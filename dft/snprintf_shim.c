/* Work-around for a false-positive _FORTIFY_SOURCE abort ("*** buffer overflow
 * detected ***") raised inside __snprintf_chk by the Ubuntu 24.04 build of
 * Quantum ESPRESSO 6.7 (pw.x), which passes an object size smaller than the
 * length argument. The replacement writes at most min(maxlen, objsize) bytes,
 * so it never writes past the object, and does not abort. Load with LD_PRELOAD.
 * Build: gcc -shared -fPIC -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -o libsnp.so snprintf_shim.c
 * (fortification must be off, or vsnprintf is redirected back to __vsnprintf_chk). */
#include <stdarg.h>
#include <stdio.h>
#include <stddef.h>

int __vsnprintf_chk(char *s, size_t maxlen, int flag, size_t slen, const char *fmt, va_list ap)
{
    (void)flag;
    size_t n = maxlen < slen ? maxlen : slen;
    return vsnprintf(s, n, fmt, ap);
}

int __snprintf_chk(char *s, size_t maxlen, int flag, size_t slen, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int r = __vsnprintf_chk(s, maxlen, flag, slen, fmt, ap);
    va_end(ap);
    return r;
}
