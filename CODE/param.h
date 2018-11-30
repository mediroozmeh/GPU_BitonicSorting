#ifdef LARGE
#define LLS 256
#define GGS 4096
#else
#define LLS 256   //// The LOCAL_SIZE_LIMIT should be set in .cl file      
#define GGS 8192*16
#endif
