// -----------------------------------------------------------------------------------------
// RTGMC demo - Linux counterparts of two upstream translation units
// -----------------------------------------------------------------------------------------
//
// The vendored tree carries only what the Windows build of this demo needs. Two upstream
// .cpp files are Linux-only in practice, so they were never pulled in, and the Linux link
// then reports them as undefined:
//
//   rgy_resource.cpp   getEmbeddedResource() reads the NNEDI weights out of a PE resource.
//                      Only the non-Windows branch of NVEncFilterNnedi::readWeights() calls
//                      it, and only when no weight file path was given. RtgmDif always
//                      passes one (nnedi3_weights.bin next to the module, the same file the
//                      Windows build loads through nnediDefaultWeightFilePath()), so this
//                      reports "no embedded resource" - the caller then takes exactly the
//                      error path Windows takes for a missing weights file.
//
//   rgy_ini.cpp        GetPrivateProfileIntCP()/GetPrivateProfileStringCP() are the Linux
//                      spellings of the Win32 profile API, which the Windows build gets as
//                      macros straight from the OS (see rgy_ini.h). Their only caller is the
//                      AFS option loader in rgy_prm.cpp, i.e. the command line front end this
//                      module does not have - RTGMC takes its parameters through RtgmDifConfig,
//                      never through an .ini file. Returning the caller's defaults is exactly what
//                      the Win32 API does when the file or the key is absent, so the semantics
//                      stay the same rather than silently inventing values.
//
// Nothing here is reachable at runtime in this demo; both exist so the link closes and the
// three-argument signature stays honest. The whole file is compiled out on Windows, where
// rgy_ini.h supplies the two as macros and nothing calls getEmbeddedResource().
// -----------------------------------------------------------------------------------------

#if !defined(_WIN32) && !defined(_WIN64)

#include <algorithm>
#include <cstring>

#include "rgy_osdep.h"
#include "rgy_tchar.h"
#include "rgy_resource.h"
#include "rgy_ini.h"

// rgy_resource.cpp

int getEmbeddedResource(void **data, const TCHAR *name, const TCHAR *type, HMODULE hModule) {
    (void)name;
    (void)type;
    (void)hModule;
    if (data != nullptr) {
        *data = nullptr;
    }
    return 0; // no embedded resource -> size 0, data untouched
}

// rgy_ini.cpp
//
// The declarations in rgy_ini.h carry the default argument (codecpage = CP_THREAD_ACP), so
// the definitions must not repeat it.

uint32_t GetPrivateProfileStringCP(const TCHAR* Section, const TCHAR* Key, const TCHAR* Default,
    TCHAR* buf, size_t nSize, const TCHAR* IniFile, uint32_t codecpage) {
    (void)Section;
    (void)Key;
    (void)IniFile;
    (void)codecpage;
    if (buf == nullptr || nSize == 0) {
        return 0;
    }
    const char *src = (Default != nullptr) ? Default : "";
    const size_t len = std::min(strlen(src), nSize - 1);
    memcpy(buf, src, len);
    buf[len] = '\0';
    return (uint32_t)len;
}

uint32_t GetPrivateProfileIntCP(const TCHAR* Section, const TCHAR* Key, const uint32_t defaultValue,
    const TCHAR* IniFile, uint32_t codecpage) {
    (void)Section;
    (void)Key;
    (void)IniFile;
    (void)codecpage;
    return defaultValue;
}

#endif // !defined(_WIN32) && !defined(_WIN64)
