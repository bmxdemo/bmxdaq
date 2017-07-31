#pragma once

// Like printf, but automatically adds a new line and increment line count
void tprintfn(const char* fmt, ...);

// return up line count lines
void treturn(int n = 0);
