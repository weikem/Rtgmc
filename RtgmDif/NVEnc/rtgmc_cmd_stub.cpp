// -----------------------------------------------------------------------------------------
// RTGMC demo - command line parser stubs
// -----------------------------------------------------------------------------------------
//
// NVEncFilterParam.cpp (which must be linked for NVEncFilterParamResize's vtable/typeinfo,
// used by NVEncFilterResize.cu) defines parse_one_vppnv_option() and reports its errors via
// print_cmd_error_invalid_value() from rgy_cmd.cpp.
//
// Pulling in rgy_cmd.cpp would drag the whole ~900 KB NVEnc command line parser (and the
// input/output modules behind it) into this demo, which never parses --vpp-* options at all.
// These stubs keep the link closed without that baggage. They are never reached at runtime
// unless somebody calls parse_one_vppnv_option() with invalid input.
// -----------------------------------------------------------------------------------------

#include "rgy_cmd.h"

static void rtgmc_cmd_stub_print(const tstring &option, const tstring &value) {
    _ftprintf(stderr, _T("[rtgmc-demo] invalid option value: %s = %s\n"), option.c_str(), value.c_str());
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue) {
    rtgmc_cmd_stub_print(strOptionName, strErrorValue);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, tstring strErrorMessage) {
    _ftprintf(stderr, _T("[rtgmc-demo] invalid option value: %s = %s (%s)\n"),
        strOptionName.c_str(), strErrorValue.c_str(), strErrorMessage.c_str());
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue,
    const std::vector<std::pair<RGY_CODEC, const CX_DESC *>> &codec_list) {
    UNREFERENCED_PARAMETER(codec_list);
    rtgmc_cmd_stub_print(strOptionName, strErrorValue);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const std::vector<tstring> &valueList) {
    UNREFERENCED_PARAMETER(valueList);
    rtgmc_cmd_stub_print(strOptionName, strErrorValue);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const CX_DESC *list) {
    UNREFERENCED_PARAMETER(list);
    rtgmc_cmd_stub_print(strOptionName, strErrorValue);
}

void print_cmd_error_invalid_value(tstring strOptionName, tstring strErrorValue, const FEATURE_DESC *list) {
    UNREFERENCED_PARAMETER(list);
    rtgmc_cmd_stub_print(strOptionName, strErrorValue);
}
