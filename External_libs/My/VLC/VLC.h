#ifdef _WIN32
#include <windows.h>
#include <process.h>    // _spawnl
#endif

namespace VLC
{
    void play(const std::string& filepath)
    {
#ifdef _WIN32
        std::string fixedPath = filepath;
        for (char& c : fixedPath)
        {
            if (c == '/')
                c = '\\';
        }

        DWORD attrs = GetFileAttributesA(fixedPath.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES)
        {
            std::cerr << "play_using_vlc: file does not exist: " << fixedPath << "\n";
            return;
        }

        const char* vlcPath64 = "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe";
        intptr_t rc = _spawnl(
            _P_NOWAIT,
            vlcPath64,
            "vlc.exe",
            "--play-and-exit",
            fixedPath.c_str(),
            nullptr
        );

        if (rc == -1)
        {
            const char* vlcPath32 = "C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe";
            rc = _spawnl(
                _P_NOWAIT,
                vlcPath32,
                "vlc.exe",
                "--play-and-exit",
                fixedPath.c_str(),
                nullptr
            );

            if (rc == -1)
            {
                std::cerr << "play_using_vlc: failed to launch VLC. "
                    << "Check that VLC is installed in Program Files / Program Files (x86).\n";
            }
        }
#else
        (void)filepath;
#endif
    }
}