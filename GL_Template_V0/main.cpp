#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// -----------------------------------------------------------------------------
// Experiment configuration
// -----------------------------------------------------------------------------
constexpr int SHAPE_GRID_WIDTH = 320;
constexpr int SHAPE_GRID_HEIGHT = 180;
constexpr int VIDEO_WIDTH = 1920;
constexpr int VIDEO_HEIGHT = 1080;
constexpr int VIDEO_FRAMES_PER_SECOND = 60;
constexpr int INITIAL_RECTANGLE_WIDTH = 48;
constexpr int INITIAL_RECTANGLE_HEIGHT = 32;
constexpr int FORBIDDEN_BORDER_MARGIN = 8;
constexpr uint32_t RANDOM_SEED = 100;
constexpr bool SHOW_PREVIEW_WINDOW = true;
constexpr bool RUN_STARTUP_SELF_TESTS = true;
constexpr bool ENABLE_DEBUG_TOPOLOGY_VALIDATION = true;
constexpr int GLOBAL_TOPOLOGY_VALIDATION_INTERVAL = 500;
constexpr int DEFAULT_MAX_ACCEPTED_MOVES_PER_GOAL = 160000;
constexpr int MAX_CONSECUTIVE_REJECTED_MOVES = 20000;
constexpr int CANDIDATES_TESTED_PER_ITERATION = 256;
constexpr double AREA_ERROR_WEIGHT = 1.0;
constexpr double PERIMETER_ERROR_WEIGHT = 1.0;
constexpr double INITIAL_ANNEALING_TEMPERATURE = 0.02;
constexpr double MINIMUM_ANNEALING_TEMPERATURE = 0.0001;
constexpr double STALL_TEMPERATURE_BOOST = 1.5;
constexpr int WITNESS_GUIDANCE_STALL_MOVES = 1200;
constexpr double WITNESS_GUIDANCE_WEIGHT = 0.035;
constexpr int AREA_GOAL_TOLERANCE = 0;
constexpr int PERIMETER_GOAL_TOLERANCE = 0;
constexpr double CAPTURE_PROGRESS_STEP = 0.02;
constexpr int CAPTURE_MINIMUM_ACCEPTED_MOVE_GAP = 4;
constexpr int CAPTURE_MAXIMUM_ACCEPTED_MOVE_GAP = 250;
constexpr int INITIAL_HOLD_FRAMES = 60;
constexpr int GOAL_REACHED_HOLD_FRAMES = 30;
constexpr int FINAL_HOLD_FRAMES = 120;
constexpr const char* OUTPUT_DIRECTORY = "output";
constexpr const char* OUTPUT_VIDEO_PATH = "output/shape_coastline_optimization.mp4";
constexpr int FFMPEG_CONSTANT_RATE_FACTOR = 18;
constexpr const char* FFMPEG_PRESET = "medium";
constexpr bool USE_GPU_CANDIDATE_SCORING = true;

// -----------------------------------------------------------------------------
// High-level artistic controls
// -----------------------------------------------------------------------------
// surface_coverage:
//     0.0 = one occupied cell (the shape is never allowed to become empty)
//     1.0 = the complete region inside FORBIDDEN_BORDER_MARGIN
//
// coast_complexity:
//     0.0 = the compact, minimum-coast witness constructed for this exact area
//     1.0 = the most rugged valid witness found for this exact area by the
//           deterministic calibration search
//
// coastline_search_steps controls how thoroughly the program searches for the
// rugged end of the range. It changes calibration quality, not the requested
// area. Higher values can discover a larger maximum coastline.
//
// visible_optimizer_move_budget controls how long the visible animation is
// allowed to search for the already-proven reachable target measurements.
struct ArtisticGoalConfig
{
    const char* name;
    double surface_coverage;
    double coast_complexity;
    int coastline_search_steps;
    int visible_optimizer_move_budget;
    int hold_frames;
};

constexpr ArtisticGoalConfig shape_goal(
    const char* name,
    double surface_coverage,
    double coast_complexity,
    int coastline_search_steps,
    int visible_optimizer_move_budget = DEFAULT_MAX_ACCEPTED_MOVES_PER_GOAL,
    int hold_frames = GOAL_REACHED_HOLD_FRAMES)
{
    return { name, surface_coverage, coast_complexity, coastline_search_steps,
            visible_optimizer_move_budget, hold_frames };
}

// Ordinary use: edit only name, surface coverage, coast complexity, and search
// steps. The last two arguments are optional advanced controls.
constexpr std::array<ArtisticGoalConfig, 6> ARTISTIC_GOAL_SEQUENCE = { {
    shape_goal("Compact expansion",     0.08, 0.05,  5000),
    shape_goal("Large calm body",      0.14, 0.12,  7000),
    shape_goal("Coastline eruption",   0.14, 0.82, 14000, 240000, 45),
    shape_goal("Partial smoothing",    0.14, 0.38, 12000, 220000),
    shape_goal("Small rugged remnant", 0.07, 0.75, 10000, 200000, 45),
    shape_goal("Final compact form",   0.04, 0.08,  6000, 160000, 60),
} };

constexpr int WITNESS_SEARCH_RESTARTS = 3;
constexpr int WITNESS_CANDIDATES_PER_STEP = 320;
constexpr int WITNESS_STALL_EXPLORATION_AFTER = 450;
constexpr double WITNESS_BASE_TEMPERATURE = 1.35;
constexpr double WITNESS_STALLED_TEMPERATURE = 4.0;
constexpr int DX4[4] = { 1, -1, 0, 0 }, DY4[4] = { 0, 0, 1, -1 };
constexpr int DX8[8] = { 1, -1, 0, 0, 1, 1, -1, -1 }, DY8[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

struct DenseIndexSet
{
    std::vector<int> entries, positions;
    explicit DenseIndexSet(int count = 0) : positions(count, -1)
    {
    }
    bool contains(int i) const
    {
        return positions.at(i) >= 0;
    }
    void set(int i, bool present)
    {
        int p = positions.at(i);
        if (present && p < 0)
        {
            positions[i] = int(entries.size());
            entries.push_back(i);
        }
        else if (!present && p >= 0)
        {
            int last = entries.back();
            entries[p] = last;
            positions[last] = p;
            entries.pop_back();
            positions[i] = -1;
        }
    }
    void clear()
    {
        std::fill(positions.begin(), positions.end(), -1);
        entries.clear();
    }
};
struct BinaryShapeState
{
    int width, height, forbidden_border_margin, occupied_area = 0, coastline_length = 0;
    std::vector<uint8_t> occupancy;
    DenseIndexSet addition_candidates, removal_candidates;
    std::vector<int64_t> last_changed_at_move;
    BinaryShapeState(int w, int h, int m)
        : width(w), height(h), forbidden_border_margin(m), occupancy(size_t(w)* h), addition_candidates(w* h),
        removal_candidates(w* h), last_changed_at_move(size_t(w)* h, -1000000)
    {
    }
    int index(int x, int y) const
    {
        assert(x >= 0 && x < width && y >= 0 && y < height);
        return y * width + x;
    }
    bool inside(int x, int y) const
    {
        return x >= 0 && x < width && y >= 0 && y < height;
    }
    bool allowed(int x, int y) const
    {
        return x >= forbidden_border_margin && x < width - forbidden_border_margin && y >= forbidden_border_margin &&
            y < height - forbidden_border_margin;
    }
    bool occupied(int x, int y) const
    {
        return inside(x, y) && occupancy[index(x, y)] != 0;
    }
};
struct ShapeGoal
{
    int target_area = 0;
    int target_perimeter = 0;
    int area_tolerance = 0;
    int perimeter_tolerance = 0;
    int calibrated_min_perimeter = 0;
    int calibrated_max_perimeter = 0;
    int visible_optimizer_move_budget = DEFAULT_MAX_ACCEPTED_MOVES_PER_GOAL;
    int hold_frames = GOAL_REACHED_HOLD_FRAMES;
    double requested_surface_coverage = 0.0;
    double requested_coast_complexity = 0.0;
    std::string descriptive_name;
    std::vector<uint8_t> witness_mask;
};
struct Validation
{
    bool valid = false, foreground_connected = false, no_holes = false, margin_clear = false;
    int area = 0, perimeter = 0;
    std::string message;
};

int cardinal_neighbors(const BinaryShapeState& s, int x, int y)
{
    int n = 0;
    for (int d = 0; d < 4; ++d)
        n += s.occupied(x + DX4[d], y + DY4[d]);
    return n;
}
bool should_add(const BinaryShapeState& s, int i)
{
    int x = i % s.width, y = i / s.width;
    return !s.occupancy[i] && s.allowed(x, y) && cardinal_neighbors(s, x, y) > 0;
}
bool should_remove(const BinaryShapeState& s, int i)
{
    int x = i % s.width, y = i / s.width;
    return s.occupancy[i] && s.occupied_area > 1 && cardinal_neighbors(s, x, y) < 4;
}
void update_frontier_cell(BinaryShapeState& s, int x, int y)
{
    if (!s.inside(x, y))
        return;
    int i = s.index(x, y);
    s.addition_candidates.set(i, should_add(s, i));
    s.removal_candidates.set(i, should_remove(s, i));
}
void update_frontier_near(BinaryShapeState& s, int i)
{
    int x = i % s.width, y = i / s.width;
    update_frontier_cell(s, x, y);
    for (int d = 0; d < 4; ++d)
        update_frontier_cell(s, x + DX4[d], y + DY4[d]);
}
void rebuild_frontier(BinaryShapeState& s)
{
    s.addition_candidates.clear();
    s.removal_candidates.clear();
    for (int i = 0; i < s.width * s.height; ++i)
    {
        s.addition_candidates.set(i, should_add(s, i));
        s.removal_candidates.set(i, should_remove(s, i));
    }
}
std::pair<int, int> measure(const BinaryShapeState& s)
{
    int a = 0, p = 0;
    for (int y = 0; y < s.height; ++y)
        for (int x = 0; x < s.width; ++x)
            if (s.occupied(x, y))
            {
                ++a;
                for (int d = 0; d < 4; ++d)
                    p += !s.occupied(x + DX4[d], y + DY4[d]);
            }
    return { a, p };
}
int local_components(const std::array<uint8_t, 9>& c, uint8_t value, bool eight)
{
    std::array<uint8_t, 9> v{};
    int count = 0;
    for (int start = 0; start < 9; ++start)
        if (!v[start] && c[start] == value)
        {
            ++count;
            int q[9], b = 0, e = 0;
            q[e++] = start;
            v[start] = 1;
            while (b < e)
            {
                int cur = q[b++], x = cur % 3, y = cur / 3;
                for (int d = 0; d < (eight ? 8 : 4); ++d)
                {
                    int nx = x + DX8[d], ny = y + DY8[d];
                    if (nx < 0 || nx >= 3 || ny < 0 || ny >= 3)
                        continue;
                    int next = ny * 3 + nx;
                    if (!v[next] && c[next] == value)
                    {
                        v[next] = 1;
                        q[e++] = next;
                    }
                }
            }
        }
    return count;
}
bool is_simple_topology_toggle(const BinaryShapeState& s, int i, bool adding)
{
    if (adding == bool(s.occupancy[i]))
        return false;
    int cx = i % s.width, cy = i / s.width;
    std::array<uint8_t, 9> after{};
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            after[(y + 1) * 3 + x + 1] = s.occupied(cx + x, cy + y);
    after[4] = adding;
    return local_components(after, 1, false) == 1 && local_components(after, 0, true) == 1;
}
bool is_valid_foreground_addition(const BinaryShapeState& s, int i)
{
    if (i < 0 || i >= s.width * s.height || s.occupancy[i])
        return false;
    int x = i % s.width, y = i / s.width;
    return s.allowed(x, y) && cardinal_neighbors(s, x, y) > 0 && is_simple_topology_toggle(s, i, true);
}
bool is_valid_foreground_removal(const BinaryShapeState& s, int i)
{
    return i >= 0 && i < s.width * s.height && s.occupancy[i] && s.occupied_area > 1 &&
        is_simple_topology_toggle(s, i, false);
}

Validation validate_complete_shape_topology(const BinaryShapeState& s)
{
    Validation r;
    auto m = measure(s);
    r.area = m.first;
    r.perimeter = m.second;
    r.margin_clear = true;
    for (int y = 0; y < s.height; ++y)
        for (int x = 0; x < s.width; ++x)
            if (s.occupied(x, y) && !s.allowed(x, y))
                r.margin_clear = false;
    std::vector<uint8_t> v(s.occupancy.size());
    std::queue<int> q;
    int start = -1;
    for (int i = 0; i < s.width * s.height; ++i)
        if (s.occupancy[i])
        {
            start = i;
            break;
        }
    int seen = 0;
    if (start >= 0)
    {
        q.push(start);
        v[start] = 1;
    }
    while (!q.empty())
    {
        int cur = q.front();
        q.pop();
        ++seen;
        int x = cur % s.width, y = cur / s.width;
        for (int d = 0; d < 4; ++d)
        {
            int nx = x + DX4[d], ny = y + DY4[d];
            if (!s.inside(nx, ny))
                continue;
            int next = s.index(nx, ny);
            if (!v[next] && s.occupancy[next])
            {
                v[next] = 1;
                q.push(next);
            }
        }
    }
    r.foreground_connected = m.first > 0 && seen == m.first;
    std::fill(v.begin(), v.end(), 0);
    start = -1;
    for (int x = 0; x < s.width && start < 0; ++x)
    {
        if (!s.occupied(x, 0))
            start = s.index(x, 0);
        else if (!s.occupied(x, s.height - 1))
            start = s.index(x, s.height - 1);
    }
    seen = 0;
    if (start >= 0)
    {
        q.push(start);
        v[start] = 1;
    }
    while (!q.empty())
    {
        int cur = q.front();
        q.pop();
        ++seen;
        int x = cur % s.width, y = cur / s.width;
        for (int d = 0; d < 8; ++d)
        {
            int nx = x + DX8[d], ny = y + DY8[d];
            if (!s.inside(nx, ny))
                continue;
            int next = s.index(nx, ny);
            if (!v[next] && !s.occupancy[next])
            {
                v[next] = 1;
                q.push(next);
            }
        }
    }
    r.no_holes = seen == s.width * s.height - m.first;
    bool cache = m.first == s.occupied_area && m.second == s.coastline_length;
    r.valid = r.foreground_connected && r.no_holes && r.margin_clear && cache;
    if (!r.valid)
    {
        std::ostringstream text;
        text << "foreground=" << r.foreground_connected << ", no_holes=" << r.no_holes << ", margin=" << r.margin_clear
            << ", area=" << s.occupied_area << '/' << m.first << ", perimeter=" << s.coastline_length << '/'
            << m.second;
        r.message = text.str();
    }
    return r;
}
bool validate_frontier(const BinaryShapeState& s)
{
    for (int i = 0; i < s.width * s.height; ++i)
        if (s.addition_candidates.contains(i) != should_add(s, i) ||
            s.removal_candidates.contains(i) != should_remove(s, i))
            return false;
    return true;
}
void initialize_rectangle(BinaryShapeState& s, int w, int h)
{
    std::fill(s.occupancy.begin(), s.occupancy.end(), 0);
    int left = (s.width - w) / 2, bottom = (s.height - h) / 2;
    for (int y = bottom; y < bottom + h; ++y)
        for (int x = left; x < left + w; ++x)
            s.occupancy[s.index(x, y)] = 1;
    auto m = measure(s);
    s.occupied_area = m.first;
    s.coastline_length = m.second;
    rebuild_frontier(s);
}
void assign_mask(BinaryShapeState& s, const std::vector<uint8_t>& mask)
{
    s.occupancy = mask;
    auto m = measure(s);
    s.occupied_area = m.first;
    s.coastline_length = m.second;
    rebuild_frontier(s);
}
int perimeter_delta(const BinaryShapeState& s, int i, bool adding)
{
    int n = cardinal_neighbors(s, i % s.width, i / s.width);
    return adding ? 4 - 2 * n : 2 * n - 4;
}
void apply_toggle(BinaryShapeState& s, int i, bool adding, int64_t move)
{
    int dp = perimeter_delta(s, i, adding);
    s.occupancy[i] = adding;
    s.occupied_area += adding ? 1 : -1;
    s.coastline_length += dp;
    s.last_changed_at_move[i] = move;
    update_frontier_near(s, i);
}

struct CandidateMove
{
    enum class Type
    {
        None,
        Add,
        Remove,
        Swap
    } type = Type::None;
    int first = -1, second = -1, area_delta = 0, perimeter_delta = 0, mismatch_delta = 0;
    double energy = std::numeric_limits<double>::infinity(), augmented_energy = std::numeric_limits<double>::infinity();
};
double goal_energy(int area, int perimeter, const ShapeGoal& g)
{
    double as = std::max(64.0, double(g.target_area)), ps = std::max(32.0, double(g.target_perimeter)),
        ae = (area - g.target_area) / as, pe = (perimeter - g.target_perimeter) / ps;
    return AREA_ERROR_WEIGHT * ae * ae + PERIMETER_ERROR_WEIGHT * pe * pe;
}
bool goal_reached(const BinaryShapeState& s, const ShapeGoal& g)
{
    return std::abs(s.occupied_area - g.target_area) <= g.area_tolerance &&
        std::abs(s.coastline_length - g.target_perimeter) <= g.perimeter_tolerance;
}
int witness_mismatch(const BinaryShapeState& s, const ShapeGoal& g)
{
    int n = 0;
    for (size_t i = 0; i < s.occupancy.size(); ++i)
        n += s.occupancy[i] != g.witness_mask[i];
    return n;
}
int mismatch_delta(const BinaryShapeState& s, const ShapeGoal& g, int i, bool adding)
{
    return int(uint8_t(adding) != g.witness_mask[i]) - int(s.occupancy[i] != g.witness_mask[i]);
}
CandidateMove evaluate_single(const BinaryShapeState& s, const ShapeGoal& g, int i, bool adding, int mismatch,
    double weight)
{
    CandidateMove m;
    if (adding ? !is_valid_foreground_addition(s, i) : !is_valid_foreground_removal(s, i))
        return m;
    m.type = adding ? CandidateMove::Type::Add : CandidateMove::Type::Remove;
    m.first = i;
    m.area_delta = adding ? 1 : -1;
    m.perimeter_delta = perimeter_delta(s, i, adding);
    m.mismatch_delta = mismatch_delta(s, g, i, adding);
    m.energy = goal_energy(s.occupied_area + m.area_delta, s.coastline_length + m.perimeter_delta, g);
    m.augmented_energy = m.energy + weight * double(mismatch + m.mismatch_delta) / s.occupancy.size();
    return m;
}
CandidateMove evaluate_swap(BinaryShapeState& s, const ShapeGoal& g, int remove_i, int add_i, int mismatch,
    double weight)
{
    CandidateMove m;
    if (remove_i == add_i || !is_valid_foreground_removal(s, remove_i))
        return m;
    int rdp = perimeter_delta(s, remove_i, false);
    s.occupancy[remove_i] = 0;
    bool valid = is_valid_foreground_addition(s, add_i);
    int adp = valid ? perimeter_delta(s, add_i, true) : 0;
    s.occupancy[remove_i] = 1;
    if (!valid)
        return m;
    m.type = CandidateMove::Type::Swap;
    m.first = remove_i;
    m.second = add_i;
    m.perimeter_delta = rdp + adp;
    m.mismatch_delta = mismatch_delta(s, g, remove_i, false) + mismatch_delta(s, g, add_i, true);
    m.energy = goal_energy(s.occupied_area, s.coastline_length + m.perimeter_delta, g);
    m.augmented_energy = m.energy + weight * double(mismatch + m.mismatch_delta) / s.occupancy.size();
    return m;
}
template <class Rng> int random_entry(const DenseIndexSet& set, Rng& rng)
{
    if (set.entries.empty())
        return -1;
    std::uniform_int_distribution<size_t> pick(0, set.entries.size() - 1);
    return set.entries[pick(rng)];
}
template <class Rng>
void sample_moves(BinaryShapeState& s, const ShapeGoal& g, Rng& rng, int mismatch, double weight, int count,
    std::vector<CandidateMove>& out)
{
    out.clear();
    std::uniform_real_distribution<double> unit(0, 1);
    int difference = g.target_area - s.occupied_area;
    for (int n = 0; n < count; ++n)
    {
        double choice = unit(rng);
        CandidateMove m;
        bool add = difference > 0 ? choice < .72 : difference < 0 ? choice >= .94 : choice >= .82 && choice < .91;
        bool remove = difference < 0 ? choice < .72 : difference > 0 ? choice >= .94 : choice >= .91;
        if (add)
        {
            int i = random_entry(s.addition_candidates, rng);
            if (i >= 0)
                m = evaluate_single(s, g, i, true, mismatch, weight);
        }
        else if (remove)
        {
            int i = random_entry(s.removal_candidates, rng);
            if (i >= 0)
                m = evaluate_single(s, g, i, false, mismatch, weight);
        }
        else
        {
            int r = random_entry(s.removal_candidates, rng), a = random_entry(s.addition_candidates, rng);
            if (r >= 0 && a >= 0)
                m = evaluate_swap(s, g, r, a, mismatch, weight);
        }
        if (m.type != CandidateMove::Type::None)
            out.push_back(m);
    }
}
bool apply_move(BinaryShapeState& s, const CandidateMove& m, int64_t number, bool validate_swap)
{
    if (m.type == CandidateMove::Type::Add)
    {
        if (!is_valid_foreground_addition(s, m.first))
            return false;
        apply_toggle(s, m.first, true, number);
    }
    else if (m.type == CandidateMove::Type::Remove)
    {
        if (!is_valid_foreground_removal(s, m.first))
            return false;
        apply_toggle(s, m.first, false, number);
    }
    else if (m.type == CandidateMove::Type::Swap)
    {
        if (!is_valid_foreground_removal(s, m.first))
            return false;
        apply_toggle(s, m.first, false, number);
        if (!is_valid_foreground_addition(s, m.second))
        {
            apply_toggle(s, m.first, true, number);
            return false;
        }
        apply_toggle(s, m.second, true, number);
        if (validate_swap && !validate_complete_shape_topology(s).valid)
        {
            apply_toggle(s, m.second, false, number);
            apply_toggle(s, m.first, true, number);
            return false;
        }
    }
    else
        return false;
    return true;
}

int usable_grid_width()
{
    return SHAPE_GRID_WIDTH - 2 * FORBIDDEN_BORDER_MARGIN;
}
int usable_grid_height()
{
    return SHAPE_GRID_HEIGHT - 2 * FORBIDDEN_BORDER_MARGIN;
}
int usable_grid_area()
{
    return usable_grid_width() * usable_grid_height();
}
int area_from_surface_coverage(double surface_coverage)
{
    double clamped = std::clamp(surface_coverage, 0.0, 1.0);
    return std::clamp(int(std::llround(clamped * usable_grid_area())), 1, usable_grid_area());
}

BinaryShapeState make_compact_shape_for_exact_area(int target_area)
{
    BinaryShapeState result(SHAPE_GRID_WIDTH, SHAPE_GRID_HEIGHT, FORBIDDEN_BORDER_MARGIN);
    target_area = std::clamp(target_area, 1, usable_grid_area());

    int best_width = 1;
    int best_height = target_area;
    int best_perimeter = std::numeric_limits<int>::max();
    double best_aspect_error = std::numeric_limits<double>::infinity();
    const double desired_aspect = double(usable_grid_width()) / usable_grid_height();

    for (int width = 1; width <= usable_grid_width(); ++width)
    {
        int height = (target_area + width - 1) / width;
        if (height > usable_grid_height())
            continue;
        int perimeter = 2 * (width + height);
        double aspect = double(width) / height;
        double aspect_error = std::abs(std::log(aspect / desired_aspect));
        if (perimeter < best_perimeter ||
            (perimeter == best_perimeter && aspect_error < best_aspect_error))
        {
            best_width = width;
            best_height = height;
            best_perimeter = perimeter;
            best_aspect_error = aspect_error;
        }
    }

    int left = FORBIDDEN_BORDER_MARGIN + (usable_grid_width() - best_width) / 2;
    int bottom = FORBIDDEN_BORDER_MARGIN + (usable_grid_height() - best_height) / 2;
    int full_rows = target_area / best_width;
    int remainder = target_area % best_width;

    for (int row = 0; row < full_rows; ++row)
        for (int x = 0; x < best_width; ++x)
            result.occupancy[result.index(left + x, bottom + row)] = 1;

    if (remainder > 0)
    {
        int partial_left = left + (best_width - remainder) / 2;
        for (int x = 0; x < remainder; ++x)
            result.occupancy[result.index(partial_left + x, bottom + full_rows)] = 1;
    }

    auto measurements = measure(result);
    result.occupied_area = measurements.first;
    result.coastline_length = measurements.second;
    rebuild_frontier(result);
    return result;
}

CandidateMove evaluate_perimeter_swap(BinaryShapeState& shape, int remove_index, int add_index)
{
    CandidateMove move;
    if (remove_index == add_index || !is_valid_foreground_removal(shape, remove_index))
        return move;

    int removal_perimeter_delta = perimeter_delta(shape, remove_index, false);
    shape.occupancy[remove_index] = 0;
    bool valid_addition = is_valid_foreground_addition(shape, add_index);
    int addition_perimeter_delta = valid_addition ? perimeter_delta(shape, add_index, true) : 0;
    shape.occupancy[remove_index] = 1;

    if (!valid_addition)
        return move;

    move.type = CandidateMove::Type::Swap;
    move.first = remove_index;
    move.second = add_index;
    move.area_delta = 0;
    move.perimeter_delta = removal_perimeter_delta + addition_perimeter_delta;
    return move;
}

struct RuggednessSearchResult
{
    int maximum_perimeter = 0;
    int selected_perimeter = 0;
    int accepted_moves = 0;
    std::vector<uint8_t> maximum_mask;
    std::vector<uint8_t> selected_mask;
};

RuggednessSearchResult run_ruggedness_search(const BinaryShapeState& compact_shape, int search_steps, uint32_t seed,
    int perimeter_to_observe)
{
    BinaryShapeState shape = compact_shape;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<CandidateMove> candidates;
    candidates.reserve(WITNESS_CANDIDATES_PER_STEP);

    RuggednessSearchResult result;
    result.maximum_perimeter = shape.coastline_length;
    result.selected_perimeter = shape.coastline_length;
    result.maximum_mask = shape.occupancy;
    result.selected_mask = shape.occupancy;
    int closest_distance = perimeter_to_observe >= 0
        ? std::abs(shape.coastline_length - perimeter_to_observe)
        : std::numeric_limits<int>::max();
    int steps_without_new_maximum = 0;

    for (int step = 0; step < std::max(0, search_steps); ++step)
    {
        candidates.clear();
        for (int sample = 0; sample < WITNESS_CANDIDATES_PER_STEP; ++sample)
        {
            int remove_index = random_entry(shape.removal_candidates, rng);
            int add_index = random_entry(shape.addition_candidates, rng);
            if (remove_index < 0 || add_index < 0)
                continue;
            CandidateMove move = evaluate_perimeter_swap(shape, remove_index, add_index);
            if (move.type != CandidateMove::Type::None)
                candidates.push_back(move);
        }
        if (candidates.empty())
            break;

        int best_delta = std::max_element(candidates.begin(), candidates.end(), [](const CandidateMove& a,
            const CandidateMove& b) {
                return a.perimeter_delta < b.perimeter_delta;
            })->perimeter_delta;

        std::vector<size_t> shortlist;
        shortlist.reserve(candidates.size());
        bool stalled = steps_without_new_maximum >= WITNESS_STALL_EXPLORATION_AFTER;
        int allowed_drop_from_best = stalled ? 6 : 2;
        for (size_t i = 0; i < candidates.size(); ++i)
            if (candidates[i].perimeter_delta >= best_delta - allowed_drop_from_best)
                shortlist.push_back(i);

        size_t chosen_index = 0;
        if (stalled && unit(rng) < 0.22)
        {
            std::uniform_int_distribution<size_t> choose_any(0, candidates.size() - 1);
            chosen_index = choose_any(rng);
        }
        else
        {
            std::uniform_int_distribution<size_t> choose_shortlisted(0, shortlist.size() - 1);
            chosen_index = shortlist[choose_shortlisted(rng)];
        }

        const CandidateMove& chosen = candidates[chosen_index];
        double temperature = stalled ? WITNESS_STALLED_TEMPERATURE : WITNESS_BASE_TEMPERATURE;
        bool accept = chosen.perimeter_delta >= 0 ||
            unit(rng) < std::exp(double(chosen.perimeter_delta) / temperature);
        if (!accept || !apply_move(shape, chosen, result.accepted_moves + 1, true))
        {
            ++steps_without_new_maximum;
            continue;
        }
        ++result.accepted_moves;

        if (shape.coastline_length > result.maximum_perimeter)
        {
            result.maximum_perimeter = shape.coastline_length;
            result.maximum_mask = shape.occupancy;
            steps_without_new_maximum = 0;
        }
        else
        {
            ++steps_without_new_maximum;
        }

        if (perimeter_to_observe >= 0)
        {
            int distance = std::abs(shape.coastline_length - perimeter_to_observe);
            if (distance < closest_distance)
            {
                closest_distance = distance;
                result.selected_perimeter = shape.coastline_length;
                result.selected_mask = shape.occupancy;
                if (distance == 0)
                    break;
            }
        }
    }

    if (perimeter_to_observe < 0)
    {
        result.selected_perimeter = result.maximum_perimeter;
        result.selected_mask = result.maximum_mask;
    }
    return result;
}

uint32_t witness_seed(size_t goal_index, int restart_index)
{
    uint32_t value = RANDOM_SEED ^ 0x9e3779b9U;
    value ^= uint32_t(goal_index + 1) * 0x85ebca6bU;
    value ^= uint32_t(restart_index + 1) * 0xc2b2ae35U;
    return value;
}

std::vector<ShapeGoal> generate_witness_goals(const BinaryShapeState&)
{
    std::cout << "Generating normalized, guaranteed-reachable witness goals...\n";
    std::vector<ShapeGoal> goals;
    goals.reserve(ARTISTIC_GOAL_SEQUENCE.size());

    for (size_t goal_index = 0; goal_index < ARTISTIC_GOAL_SEQUENCE.size(); ++goal_index)
    {
        const ArtisticGoalConfig& control = ARTISTIC_GOAL_SEQUENCE[goal_index];
        double requested_surface = std::clamp(control.surface_coverage, 0.0, 1.0);
        double requested_complexity = std::clamp(control.coast_complexity, 0.0, 1.0);
        int target_area = area_from_surface_coverage(requested_surface);

        BinaryShapeState compact = make_compact_shape_for_exact_area(target_area);
        Validation compact_validation = validate_complete_shape_topology(compact);
        if (!compact_validation.valid)
        {
            std::cerr << "Could not construct compact witness for goal " << control.name << ": "
                << compact_validation.message << '\n';
            return {};
        }

        int minimum_perimeter = compact.coastline_length;
        RuggednessSearchResult best_calibration;
        best_calibration.maximum_perimeter = minimum_perimeter;
        best_calibration.maximum_mask = compact.occupancy;
        int winning_restart = 0;

        for (int restart = 0; restart < WITNESS_SEARCH_RESTARTS; ++restart)
        {
            RuggednessSearchResult calibration =
                run_ruggedness_search(compact, control.coastline_search_steps,
                    witness_seed(goal_index, restart), -1);
            if (calibration.maximum_perimeter > best_calibration.maximum_perimeter)
            {
                best_calibration = std::move(calibration);
                winning_restart = restart;
            }
        }

        int maximum_perimeter = best_calibration.maximum_perimeter;
        double requested_perimeter = minimum_perimeter +
            requested_complexity * double(maximum_perimeter - minimum_perimeter);
        int observed_target = int(std::llround(requested_perimeter / 2.0)) * 2;

        std::vector<uint8_t> selected_mask;
        int selected_perimeter = minimum_perimeter;
        if (requested_complexity <= 0.0 || maximum_perimeter == minimum_perimeter)
        {
            selected_mask = compact.occupancy;
        }
        else if (requested_complexity >= 1.0)
        {
            selected_mask = best_calibration.maximum_mask;
            selected_perimeter = maximum_perimeter;
        }
        else
        {
            RuggednessSearchResult selection =
                run_ruggedness_search(compact, control.coastline_search_steps,
                    witness_seed(goal_index, winning_restart), observed_target);
            selected_mask = std::move(selection.selected_mask);
            selected_perimeter = selection.selected_perimeter;
        }

        BinaryShapeState selected = compact;
        assign_mask(selected, selected_mask);
        Validation selected_validation = validate_complete_shape_topology(selected);
        if (!selected_validation.valid || selected.occupied_area != target_area)
        {
            std::cerr << "Generated witness is invalid for goal " << control.name << ": "
                << selected_validation.message << '\n';
            return {};
        }
        selected_perimeter = selected.coastline_length;

        ShapeGoal goal;
        goal.target_area = selected.occupied_area;
        goal.target_perimeter = selected_perimeter;
        goal.area_tolerance = AREA_GOAL_TOLERANCE;
        goal.perimeter_tolerance = PERIMETER_GOAL_TOLERANCE;
        goal.calibrated_min_perimeter = minimum_perimeter;
        goal.calibrated_max_perimeter = maximum_perimeter;
        goal.visible_optimizer_move_budget = std::max(1, control.visible_optimizer_move_budget);
        goal.hold_frames = std::max(0, control.hold_frames);
        goal.requested_surface_coverage = requested_surface;
        goal.requested_coast_complexity = requested_complexity;
        goal.descriptive_name = control.name;
        goal.witness_mask = std::move(selected_mask);
        goals.push_back(std::move(goal));

        double actual_surface = double(goals.back().target_area) / usable_grid_area();
        double actual_complexity = maximum_perimeter > minimum_perimeter
            ? double(goals.back().target_perimeter - minimum_perimeter) /
            double(maximum_perimeter - minimum_perimeter)
            : 0.0;
        std::cout << "  " << goals.back().descriptive_name
            << ": surface " << std::fixed << std::setprecision(3) << actual_surface
            << " (requested " << requested_surface << ")"
            << ", coast " << actual_complexity
            << " (requested " << requested_complexity << ")"
            << ", area " << goals.back().target_area
            << ", perimeter " << goals.back().target_perimeter
            << ", calibrated range [" << minimum_perimeter << ", " << maximum_perimeter << "]"
            << ", search steps " << control.coastline_search_steps << '\n';
    }
    return goals;
}

bool run_self_tests()
{
    std::cout << "Running deterministic shape self-tests...\n";
    bool ok = true;
    auto require = [&](bool value, const char* name) {
        if (!value)
            std::cerr << "SELF-TEST FAILED: " << name << '\n';
        ok &= value;
        };
    BinaryShapeState one(9, 9, 0);
    one.occupancy[one.index(4, 4)] = 1;
    one.occupied_area = 1;
    one.coastline_length = 4;
    rebuild_frontier(one);
    require(measure(one) == std::pair{ 1, 4 }, "single pixel");
    BinaryShapeState rectangle(12, 10, 0);
    initialize_rectangle(rectangle, 4, 3);
    require(rectangle.occupied_area == 12 && rectangle.coastline_length == 14, "rectangle");
    int grow = rectangle.index(8, 5);
    require(perimeter_delta(rectangle, grow, true) == 2, "add delta");
    require(perimeter_delta(rectangle, rectangle.index(4, 3), false) == 0, "remove delta");
    BinaryShapeState bridge(9, 7, 0);
    for (int x = 2; x <= 6; ++x)
        bridge.occupancy[bridge.index(x, 3)] = 1;
    bridge.occupied_area = 5;
    bridge.coastline_length = 12;
    rebuild_frontier(bridge);
    require(!is_valid_foreground_removal(bridge, bridge.index(4, 3)), "bridge rejection");
    BinaryShapeState ring(9, 9, 0);
    for (int y = 3; y <= 5; ++y)
        for (int x = 3; x <= 5; ++x)
            if (x == 3 || x == 5 || y == 3 || y == 5)
                ring.occupancy[ring.index(x, y)] = 1;
    ring.occupancy[ring.index(4, 3)] = 0;
    auto rm = measure(ring);
    ring.occupied_area = rm.first;
    ring.coastline_length = rm.second;
    rebuild_frontier(ring);
    require(!is_valid_foreground_addition(ring, ring.index(4, 3)), "hole closing rejection");
    require(is_valid_foreground_addition(rectangle, grow), "ordinary growth");
    BinaryShapeState disconnected(9, 9, 0);
    disconnected.occupancy[disconnected.index(2, 2)] = disconnected.occupancy[disconnected.index(6, 6)] = 1;
    auto dm = measure(disconnected);
    disconnected.occupied_area = dm.first;
    disconnected.coastline_length = dm.second;
    require(!validate_complete_shape_topology(disconnected).foreground_connected, "disconnected detection");
    BinaryShapeState hole(9, 9, 0);
    for (int y = 2; y <= 6; ++y)
        for (int x = 2; x <= 6; ++x)
            if (x == 2 || x == 6 || y == 2 || y == 6)
                hole.occupancy[hole.index(x, y)] = 1;
    auto hm = measure(hole);
    hole.occupied_area = hm.first;
    hole.coastline_length = hm.second;
    require(!validate_complete_shape_topology(hole).no_holes, "hole detection");
    require(validate_frontier(rectangle), "frontier");
    std::cout << (ok ? "All shape self-tests passed.\n" : "Shape self-tests failed.\n");
    return ok;
}

GLuint compile_shader(GLenum type, const char* source, const char* label)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok)
        return shader;
    GLint length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> log(std::max(1, length));
    glGetShaderInfoLog(shader, length, nullptr, log.data());
    std::cerr << "Complete " << label << " log:\n" << log.data() << '\n';
    glDeleteShader(shader);
    return 0;
}
GLuint link_program(std::initializer_list<GLuint> shaders, const char* label)
{
    GLuint program = glCreateProgram();
    for (GLuint shader : shaders)
        glAttachShader(program, shader);
    glLinkProgram(program);
    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (ok)
        return program;
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::vector<char> log(std::max(1, length));
    glGetProgramInfoLog(program, length, nullptr, log.data());
    std::cerr << "Complete " << label << " link log:\n" << log.data() << '\n';
    glDeleteProgram(program);
    return 0;
}

struct GpuCandidateScorer
{
    struct alignas(16) Input
    {
        int area, perimeter, mismatch, padding;
    };
    GLuint program = 0, input_buffer = 0, output_buffer = 0;
    std::vector<Input> inputs;
    std::vector<float> outputs;
    int capacity = 0;
    bool initialize(int maximum_candidates)
    {
        constexpr const char* source = R"GLSL(#version 460 core
layout(local_size_x=64) in;
struct CandidateInput{int area;int perimeter;int mismatch;int padding;};
layout(std430,binding=0)readonly buffer Inputs{CandidateInput values[];}inputData;
layout(std430,binding=1)writeonly buffer Outputs{float scores[];}outputData;
uniform int candidateCount;uniform int targetArea;uniform int targetPerimeter;uniform int gridArea;
uniform float areaScale;uniform float perimeterScale;uniform float areaWeight;uniform float perimeterWeight;uniform float witnessWeight;
void main(){uint i=gl_GlobalInvocationID.x;if(i>=uint(candidateCount))return;CandidateInput c=inputData.values[i];float ae=(float(c.area-targetArea)/areaScale);float pe=(float(c.perimeter-targetPerimeter)/perimeterScale);outputData.scores[i]=areaWeight*ae*ae+perimeterWeight*pe*pe+witnessWeight*float(c.mismatch)/float(gridArea);}
)GLSL";
        GLuint shader = compile_shader(GL_COMPUTE_SHADER, source, "candidate-scoring compute shader");
        if (!shader)
            return false;
        program = link_program({ shader }, "candidate-scoring compute program");
        glDeleteShader(shader);
        if (!program)
            return false;
        capacity = maximum_candidates;
        inputs.reserve(capacity);
        outputs.resize(capacity);
        glGenBuffers(1, &input_buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Input) * capacity, nullptr, GL_DYNAMIC_DRAW);
        glGenBuffers(1, &output_buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * capacity, nullptr, GL_DYNAMIC_READ);
        return true;
    }
    bool score(std::vector<CandidateMove>& moves, const BinaryShapeState& s, const ShapeGoal& g, int mismatch,
        double witness_weight)
    {
        if (!USE_GPU_CANDIDATE_SCORING)
            return true;
        if (!program || int(moves.size()) > capacity)
            return false;
        inputs.clear();
        for (const auto& m : moves)
            inputs.push_back({ s.occupied_area + m.area_delta, s.coastline_length + m.perimeter_delta,
                              mismatch + m.mismatch_delta, 0 });
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, input_buffer);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Input) * inputs.size(), inputs.data());
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, input_buffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, output_buffer);
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "candidateCount"), int(moves.size()));
        glUniform1i(glGetUniformLocation(program, "targetArea"), g.target_area);
        glUniform1i(glGetUniformLocation(program, "targetPerimeter"), g.target_perimeter);
        glUniform1i(glGetUniformLocation(program, "gridArea"), s.width * s.height);
        glUniform1f(glGetUniformLocation(program, "areaScale"), float(std::max(64.0, double(g.target_area))));
        glUniform1f(glGetUniformLocation(program, "perimeterScale"), float(std::max(32.0, double(g.target_perimeter))));
        glUniform1f(glGetUniformLocation(program, "areaWeight"), float(AREA_ERROR_WEIGHT));
        glUniform1f(glGetUniformLocation(program, "perimeterWeight"), float(PERIMETER_ERROR_WEIGHT));
        glUniform1f(glGetUniformLocation(program, "witnessWeight"), float(witness_weight));
        glDispatchCompute(GLuint((moves.size() + 63) / 64), 1, 1);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, output_buffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * moves.size(), outputs.data());
        for (size_t i = 0; i < moves.size(); ++i)
            moves[i].augmented_energy = outputs[i];
        return glGetError() == GL_NO_ERROR;
    }
    void shutdown()
    {
        if (input_buffer)
            glDeleteBuffers(1, &input_buffer);
        if (output_buffer)
            glDeleteBuffers(1, &output_buffer);
        if (program)
            glDeleteProgram(program);
        input_buffer = output_buffer = program = 0;
    }
};

struct ShapeRenderer
{
    GLFWwindow* window = nullptr;
    GLuint program = 0, vao = 0, shape_texture = 0, age_texture = 0, framebuffer = 0, color_texture = 0;
    std::vector<uint8_t> age_upload, bottom_up_pixels, top_down_pixels;
    GpuCandidateScorer scorer;
    bool initialize()
    {
        glfwSetErrorCallback([](int code, const char* description) {
            std::cerr << "GLFW error " << code << ": " << description << '\n';
            });
        if (!glfwInit())
        {
            std::cerr << "Failed to initialize GLFW.\n";
            return false;
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, SHOW_PREVIEW_WINDOW ? GLFW_TRUE : GLFW_FALSE);
        window = glfwCreateWindow(960, 540, "Shape Coastline Optimization", nullptr, nullptr);
        if (!window)
        {
            std::cerr << "Failed to create an OpenGL 4.6 core window.\n";
            return false;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);
        glewExperimental = GL_TRUE;
        GLenum status = glewInit();
        if (status != GLEW_OK)
        {
            std::cerr << "GLEW failed: " << glewGetErrorString(status) << '\n';
            return false;
        }
        glGetError();
        std::cout << "OpenGL: " << glGetString(GL_VERSION) << " | " << glGetString(GL_RENDERER) << '\n';
        constexpr const char* vertex = R"GLSL(#version 460 core
out vec2 uv;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.0-1.0,0,1);}
)GLSL";
        constexpr const char* fragment = R"GLSL(#version 460 core
in vec2 uv;out vec4 color;uniform usampler2D shapeTexture;uniform sampler2D ageTexture;uniform ivec2 gridSize;uniform vec2 contentMin;uniform vec2 contentSize;uniform float pulse;
uint cell(ivec2 p){if(any(lessThan(p,ivec2(0)))||any(greaterThanEqual(p,gridSize)))return 0u;return texelFetch(shapeTexture,p,0).r;}
void main(){vec3 dark=vec3(.008,.014,.025);vec2 local=(uv-contentMin)/contentSize;if(any(lessThan(local,vec2(0)))||any(greaterThanEqual(local,vec2(1)))){color=vec4(dark,1);return;}ivec2 p=clamp(ivec2(local*vec2(gridSize)),ivec2(0),gridSize-1);uint c=cell(p),l=cell(p+ivec2(-1,0)),r=cell(p+ivec2(1,0)),d=cell(p+ivec2(0,-1)),u=cell(p+ivec2(0,1));bool coast=c==1u&&(l==0u||r==0u||d==0u||u==0u);bool glow=c==0u&&(l==1u||r==1u||d==1u||u==1u);float recent=texelFetch(ageTexture,p,0).r;vec2 f=fract(local*vec2(gridSize));float edge=smoothstep(0.0,0.08,min(min(f.x,1-f.x),min(f.y,1-f.y)));vec3 result=dark+vec3(.005,.011,.019)*edge;if(glow)result+=vec3(.01,.09,.13);if(c==1u)result=mix(vec3(.07,.55,.72),vec3(.40,.95,.86),recent)*mix(.88,1.0,edge)+pulse*vec3(.16,.10,.03);if(coast)result=mix(result,vec3(.78,1,.96),.72);color=vec4(result,1);}
)GLSL";
        GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex, "vertex shader"),
            fs = compile_shader(GL_FRAGMENT_SHADER, fragment, "fragment shader");
        if (!vs || !fs)
            return false;
        program = link_program({ vs, fs }, "render program");
        glDeleteShader(vs);
        glDeleteShader(fs);
        if (!program)
            return false;
        glGenVertexArrays(1, &vao);
        glGenTextures(1, &shape_texture);
        glBindTexture(GL_TEXTURE_2D, shape_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, SHAPE_GRID_WIDTH, SHAPE_GRID_HEIGHT, 0, GL_RED_INTEGER,
            GL_UNSIGNED_BYTE, nullptr);
        glGenTextures(1, &age_texture);
        glBindTexture(GL_TEXTURE_2D, age_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, SHAPE_GRID_WIDTH, SHAPE_GRID_HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE,
            nullptr);
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glGenTextures(1, &color_texture);
        glBindTexture(GL_TEXTURE_2D, color_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, VIDEO_WIDTH, VIDEO_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cerr << "Video framebuffer is incomplete.\n";
            return false;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        age_upload.resize(size_t(SHAPE_GRID_WIDTH) * SHAPE_GRID_HEIGHT);
        size_t bytes = size_t(VIDEO_WIDTH) * VIDEO_HEIGHT * 4;
        bottom_up_pixels.resize(bytes);
        top_down_pixels.resize(bytes);
        return !USE_GPU_CANDIDATE_SCORING || scorer.initialize(CANDIDATES_TESTED_PER_ITERATION * 2);
    }
    bool render_and_read(const BinaryShapeState& s, int64_t move, float pulse_value)
    {
        for (size_t i = 0; i < age_upload.size(); ++i)
        {
            double value = std::exp(-double(std::max<int64_t>(0, move - s.last_changed_at_move[i])) / 45.0);
            age_upload[i] = uint8_t(std::clamp(value * 255.0, 0.0, 255.0));
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, shape_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s.width, s.height, GL_RED_INTEGER, GL_UNSIGNED_BYTE,
            s.occupancy.data());
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, age_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s.width, s.height, GL_RED, GL_UNSIGNED_BYTE, age_upload.data());
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glViewport(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT);
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "shapeTexture"), 0);
        glUniform1i(glGetUniformLocation(program, "ageTexture"), 1);
        glUniform2i(glGetUniformLocation(program, "gridSize"), s.width, s.height);
        float va = float(VIDEO_WIDTH) / VIDEO_HEIGHT, ga = float(s.width) / s.height, w = 1, h = 1;
        if (va > ga)
            w = ga / va;
        else
            h = va / ga;
        glUniform2f(glGetUniformLocation(program, "contentMin"), (1 - w) * .5f, (1 - h) * .5f);
        glUniform2f(glGetUniformLocation(program, "contentSize"), w, h);
        glUniform1f(glGetUniformLocation(program, "pulse"), pulse_value);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (SHOW_PREVIEW_WINDOW)
        {
            int pw, ph;
            glfwGetFramebufferSize(window, &pw, &ph);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBlitFramebuffer(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, 0, 0, pw, ph, GL_COLOR_BUFFER_BIT, GL_LINEAR);
            glfwSwapBuffers(window);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, VIDEO_WIDTH, VIDEO_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, bottom_up_pixels.data());
        size_t row = size_t(VIDEO_WIDTH) * 4;
        for (int y = 0; y < VIDEO_HEIGHT; ++y)
            std::copy_n(bottom_up_pixels.data() + size_t(y) * row, row,
                top_down_pixels.data() + size_t(VIDEO_HEIGHT - 1 - y) * row);
        glfwPollEvents();
        return glGetError() == GL_NO_ERROR;
    }
    bool cancelled() const
    {
        return window && (glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS);
    }
    void shutdown()
    {
        scorer.shutdown();
        if (program)
            glDeleteProgram(program);
        if (vao)
            glDeleteVertexArrays(1, &vao);
        if (shape_texture)
            glDeleteTextures(1, &shape_texture);
        if (age_texture)
            glDeleteTextures(1, &age_texture);
        if (color_texture)
            glDeleteTextures(1, &color_texture);
        if (framebuffer)
            glDeleteFramebuffers(1, &framebuffer);
        if (window)
            glfwDestroyWindow(window);
        program = vao = shape_texture = age_texture = color_texture = framebuffer = 0;
        window = nullptr;
        glfwTerminate();
    }
    ~ShapeRenderer()
    {
        shutdown();
    }
};

struct FfmpegVideoWriter
{
    FILE* pipe = nullptr;
    int exit_status = -1;
    bool failed = false;
    bool open()
    {
        std::error_code error;
        std::filesystem::create_directories(OUTPUT_DIRECTORY, error);
        if (error)
        {
            std::cerr << "Cannot create output directory: " << error.message() << '\n';
            return false;
        }
        std::ostringstream command;
        command << "ffmpeg -hide_banner -loglevel warning -y -f rawvideo -vcodec rawvideo -pix_fmt rgba -s "
            << VIDEO_WIDTH << 'x' << VIDEO_HEIGHT << " -r " << VIDEO_FRAMES_PER_SECOND
            << " -i - -an -c:v libx264 -preset " << FFMPEG_PRESET << " -crf " << FFMPEG_CONSTANT_RATE_FACTOR
            << " -pix_fmt yuv420p \"" << OUTPUT_VIDEO_PATH << "\"";
#ifdef _WIN32
        pipe = _popen(command.str().c_str(), "w");
        if (pipe && _setmode(_fileno(pipe), _O_BINARY) == -1)
        {
            std::cerr << "Cannot set FFmpeg pipe to binary mode.\n";
            _pclose(pipe);
            pipe = nullptr;
        }
#else
        pipe = popen(command.str().c_str(), "w");
#endif
        if (!pipe)
        {
            std::cerr << "Failed to launch FFmpeg: " << command.str() << '\n';
            return false;
        }
        std::cout << "Streaming raw RGBA frames directly to FFmpeg.\n";
        return true;
    }
    bool write(const std::vector<uint8_t>& pixels)
    {
        if (!pipe || failed)
            return false;
        size_t count = std::fwrite(pixels.data(), 1, pixels.size(), pipe);
        if (count != pixels.size())
        {
            std::cerr << "FFmpeg write failed after " << count << " bytes.\n";
            failed = true;
            return false;
        }
        return true;
    }
    bool close()
    {
        if (!pipe)
            return exit_status == 0;
#ifdef _WIN32
        exit_status = _pclose(pipe);
#else
        exit_status = pclose(pipe);
#endif
        pipe = nullptr;
        if (exit_status != 0)
            std::cerr << "FFmpeg exited with status " << exit_status << '\n';
        return exit_status == 0 && !failed;
    }
    ~FfmpegVideoWriter()
    {
        if (pipe)
            close();
    }
};

int main()
{
    std::cout << "Binary Shape Coastline Optimization\nGrid: " << SHAPE_GRID_WIDTH << 'x' << SHAPE_GRID_HEIGHT
        << " | Video: " << VIDEO_WIDTH << 'x' << VIDEO_HEIGHT << " @ " << VIDEO_FRAMES_PER_SECOND
        << " fps\nSeed: " << RANDOM_SEED << " | margin: " << FORBIDDEN_BORDER_MARGIN
        << " | GPU candidate scoring: " << (USE_GPU_CANDIDATE_SCORING ? "compute shader" : "CPU") << "\n";
    if (RUN_STARTUP_SELF_TESTS && !run_self_tests())
        return 1;
    ShapeRenderer renderer;
    if (!renderer.initialize())
        return 1;
    BinaryShapeState shape(SHAPE_GRID_WIDTH, SHAPE_GRID_HEIGHT, FORBIDDEN_BORDER_MARGIN);
    initialize_rectangle(shape, INITIAL_RECTANGLE_WIDTH, INITIAL_RECTANGLE_HEIGHT);
    std::vector<uint8_t> initial = shape.occupancy;
    Validation initial_validation = validate_complete_shape_topology(shape);
    if (!initial_validation.valid)
    {
        std::cerr << "Invalid initial shape: " << initial_validation.message << '\n';
        return 1;
    }
    std::vector<ShapeGoal> goals = generate_witness_goals(shape);
    if (goals.empty())
        return 1;
    assign_mask(shape, initial);
    FfmpegVideoWriter writer;
    if (!writer.open())
        return 1;
    int64_t accepted_total = 0, rejected_total = 0;
    int frames = 0, completed = 0;
    bool topology_ok = true, ok = true, cancelled = false;
    auto capture = [&](float pulse) {
        Validation v = validate_complete_shape_topology(shape);
        if (!v.valid)
        {
            std::cerr << "Capture topology failure: " << v.message << '\n';
            topology_ok = false;
            return false;
        }
        if (!renderer.render_and_read(shape, accepted_total, pulse))
        {
            std::cerr << "OpenGL frame failure.\n";
            return false;
        }
        if (!writer.write(renderer.top_down_pixels))
            return false;
        ++frames;
        return true;
        };
    for (int i = 0; i < INITIAL_HOLD_FRAMES && ok; ++i)
        ok = capture(0);
    std::mt19937 rng(RANDOM_SEED);
    std::uniform_real_distribution<double> unit(0, 1);
    std::vector<CandidateMove> candidates;
    candidates.reserve(CANDIDATES_TESTED_PER_ITERATION * 2);
    for (size_t goal_index = 0; goal_index < goals.size() && ok; ++goal_index)
    {
        const ShapeGoal& goal = goals[goal_index];
        double starting = goal_energy(shape.occupied_area, shape.coastline_length, goal), current = starting,
            best = starting, next_capture = CAPTURE_PROGRESS_STEP;
        int accepted = 0, rejected = 0, consecutive_rejected = 0, since_capture = 0, since_best = 0,
            mismatch = witness_mismatch(shape, goal);
        std::cout << "\nGoal " << goal_index + 1 << '/' << goals.size() << ": " << goal.descriptive_name
            << " | requested surface=" << goal.requested_surface_coverage
            << ", coast=" << goal.requested_coast_complexity
            << " | calibrated coast range [" << goal.calibrated_min_perimeter << ','
            << goal.calibrated_max_perimeter << "]"
            << " | current (" << shape.occupied_area << ',' << shape.coastline_length << ") target ("
            << goal.target_area << ',' << goal.target_perimeter << ")\n";
        while (!goal_reached(shape, goal) && accepted < goal.visible_optimizer_move_budget &&
            consecutive_rejected < MAX_CONSECUTIVE_REJECTED_MOVES)
        {
            if (renderer.cancelled())
            {
                cancelled = true;
                break;
            }
            bool guided = since_best >= WITNESS_GUIDANCE_STALL_MOVES;
            double weight = guided ? WITNESS_GUIDANCE_WEIGHT : 0;
            int test_count = guided ? CANDIDATES_TESTED_PER_ITERATION * 2 : CANDIDATES_TESTED_PER_ITERATION;
            sample_moves(shape, goal, rng, mismatch, weight, test_count, candidates);
            if (candidates.empty())
            {
                ++rejected;
                ++rejected_total;
                ++consecutive_rejected;
                continue;
            }
            if (!renderer.scorer.score(candidates, shape, goal, mismatch, weight))
            {
                std::cerr << "GPU candidate scoring failed.\n";
                ok = false;
                break;
            }
            auto chosen = std::min_element(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.augmented_energy < b.augmented_energy;
                });
            double progress = starting > 0 ? std::clamp(1 - best / starting, 0.0, 1.0) : 1;
            double temperature =
                std::max(MINIMUM_ANNEALING_TEMPERATURE,
                    INITIAL_ANNEALING_TEMPERATURE * std::max(.02, 1 - progress) * std::max(.02, 1 - progress));
            if (guided)
                temperature *= STALL_TEMPERATURE_BOOST;
            double current_augmented = current + weight * double(mismatch) / shape.occupancy.size();
            bool accept = chosen->augmented_energy <= current_augmented ||
                unit(rng) < std::exp(-(chosen->augmented_energy - current_augmented) / temperature);
            if (!accept || !apply_move(shape, *chosen, accepted_total + 1, true))
            {
                ++rejected;
                ++rejected_total;
                ++consecutive_rejected;
                continue;
            }
            ++accepted;
            ++accepted_total;
            ++since_capture;
            ++since_best;
            consecutive_rejected = 0;
            current = goal_energy(shape.occupied_area, shape.coastline_length, goal);
            mismatch += chosen->mismatch_delta;
            bool new_best = current < best;
            if (new_best)
            {
                best = current;
                since_best = 0;
            }
            if (ENABLE_DEBUG_TOPOLOGY_VALIDATION && accepted_total % GLOBAL_TOPOLOGY_VALIDATION_INTERVAL == 0)
            {
                Validation v = validate_complete_shape_topology(shape);
                if (!v.valid || !validate_frontier(shape))
                {
                    std::cerr << "Periodic validation failed: " << v.message << '\n';
                    topology_ok = false;
                    ok = false;
                    break;
                }
            }
            double updated = starting > 0 ? std::clamp(1 - best / starting, 0.0, 1.0) : 1;
            bool threshold = updated + 1e-12 >= next_capture;
            bool best_event = new_best && since_capture >= CAPTURE_MINIMUM_ACCEPTED_MOVE_GAP;
            bool max_gap = since_capture >= CAPTURE_MAXIMUM_ACCEPTED_MOVE_GAP;
            if (threshold || best_event || max_gap)
            {
                ok = capture(0);
                since_capture = 0;
                while (next_capture <= updated + 1e-12)
                    next_capture += CAPTURE_PROGRESS_STEP;
            }
            if (accepted % 1000 == 0)
                std::cout << "  area " << shape.occupied_area << '/' << goal.target_area << ", perimeter "
                << shape.coastline_length << '/' << goal.target_perimeter << ", energy " << std::scientific
                << current << ", best " << best << std::fixed << ", accepted " << accepted << ", rejected "
                << rejected << ", frames " << frames << ", temperature " << temperature
                << (guided ? ", witness guidance" : "") << '\n';
        }
        if (cancelled || !ok)
            break;
        if (!goal_reached(shape, goal))
        {
            std::cerr << "WARNING: goal budget exhausted; advancing.\n";
            continue;
        }
        Validation v = validate_complete_shape_topology(shape);
        if (!v.valid || v.area != goal.target_area || v.perimeter != goal.target_perimeter)
        {
            std::cerr << "Goal completion validation failed: " << v.message << '\n';
            topology_ok = false;
            ok = false;
            break;
        }
        ++completed;
        std::cout << "Reached goal " << goal_index + 1 << ": area " << shape.occupied_area << ", perimeter "
            << shape.coastline_length << ", accepted " << accepted << ", rejected " << rejected << '\n';
        for (int f = 0; f < goal.hold_frames && ok; ++f)
            ok = capture(float(.5 + .5 * std::sin(f * 6.28318530718 / std::max(1, goal.hold_frames))));
    }
    if (ok && !cancelled)
    {
        Validation v = validate_complete_shape_topology(shape);
        if (!v.valid)
        {
            topology_ok = false;
            ok = false;
        }
        for (int f = 0; f < FINAL_HOLD_FRAMES && ok; ++f)
            ok = capture(0);
    }
    bool ffmpeg_ok = writer.close();
    double duration = double(frames) / VIDEO_FRAMES_PER_SECOND;
    std::cout << "\nGeneration summary\nOutput video: " << std::filesystem::absolute(OUTPUT_VIDEO_PATH).string()
        << "\nGoals completed: " << completed << '/' << goals.size() << "\nAccepted moves: " << accepted_total
        << "\nRejected moves: " << rejected_total << "\nRendered frames: " << frames
        << "\nApproximate duration: " << std::fixed << std::setprecision(2) << duration
        << " seconds\nFinal area/perimeter: " << shape.occupied_area << '/' << shape.coastline_length
        << "\nAll topology validations passed: " << (topology_ok ? "yes" : "no")
        << "\nFFmpeg exit status: " << writer.exit_status << '\n';
    if (cancelled)
        std::cout << "Cancelled cleanly; FFmpeg was finalized.\n";
    bool exists = std::filesystem::exists(OUTPUT_VIDEO_PATH) && std::filesystem::file_size(OUTPUT_VIDEO_PATH) > 0;
    if (!exists)
        std::cerr << "Output video is missing or empty.\n";
    return ok && ffmpeg_ok && exists && topology_ok ? 0 : 1;
}
