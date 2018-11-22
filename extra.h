#define RED   "\x1B[31m"
#define GREEN   "\x1B[32m"
#define YELLOW   "\x1B[33m"
#define BLUE   "\x1B[34m"
#define RESET "\x1B[0m"




  /// Capture the elapsd time for a given event in nanosecond.// 
   
    double time_profiler(cl_event event, cl_int ret) {
             
           cl_ulong start_time=0;
           cl_ulong end_time=0;
	  

     ret =  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
                if(ret != CL_SUCCESS)
        	printf(RED "Profiling is failed\n" RESET); 

     ret =  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &end_time, NULL);          
               if(ret != CL_SUCCESS)
	       printf(RED "Profiling is failed\n" RESET);           
       
               return  1.0e-9 * ( end_time - start_time) 	; 
     
}

///
//
// PRINT OUT THE ERROR MESSAGE
print_error(char input[50], int line_number ){
	 printf(RED "%s :#%d \n "  RESET , input ,line_number - 2);

}
//////// Load source file 
int 
load_file_to_memory(char *kernelname ,char **result)
{
 

        size_t size = 0;
        FILE *f = fopen(kernelname, "rb");
        if (f == NULL) {
            *result = NULL;
            return -1; // -1 means file opening fail 
        }
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        fseek(f, 0, SEEK_SET);
        *result = (char *) malloc(size + 1);
        if (size != fread(*result, sizeof(char), size, f)) {
            free(*result);
            return -2; // -2 means file reading fail 
        }
        fclose(f);
        (*result)[size] = 0;
        return size;
}




