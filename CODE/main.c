#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#define CL_USE_DEPRECTED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif
#include "extra.h"
#include "param.h"

int main(int argc, char** argv) {

    // Create the two input vectors
    unsigned int i;
    unsigned int DATA_SIZE = GGS   ; // set the global size
    unsigned int LOCAL_SIZE = LLS   ;  

    int random;
    

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    size_t valuesize;
    char *value;
    cl_uint maxComputeUnits;
    cl_uint workdimensions;
    size_t maxWorkitem[3];
    size_t maxWorkGroup;
    const unsigned int MAX_GPU_COUNT = 8;
    cl_event myevent; 
    double kernel_time=0;
    cl_uint dir=1;
    cl_int err; 
    char *source_str;
    size_t source_size;
    size_t preferred_groupsize;
          
    char *kernelsource;
    char filename[]="BitonicSort.cl";

   
  


        FILE *fp;
        fp = fopen("output.txt","w"); 

         #ifdef options
	   char options[]="-cl-mad-enable -cl-fast-relaxed-math";
	   fprintf(fp,"The compiler options is enabled for clbuild ");
         #else
	    char options[]="";
	 #endif	
	   


       #ifdef OPTIMIZED
    fprintf(fp,"The level 0 optimization is enabled from make file :OO1,OO2,PP \n");
       #endif
     

     fprintf(fp,"GLOBAL SIZE:%d , LOCAL SIZE:%d, Total WORK GROUP %d\n", DATA_SIZE, LOCAL_SIZE/2, DATA_SIZE/LOCAL_SIZE);

       

    ////// #Regione 1: Get the Device and Platfrom info and IDs
    //
    //
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
       if(ret!=CL_SUCCESS)
       print_error("Platform is not detected", __LINE__);
       
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
            &device_id, &ret_num_devices);      
       if(ret!=CL_SUCCESS)
       print_error( "Device is not detected",  __LINE__);
       
         


    //////////////////

     clGetPlatformInfo (platform_id, CL_PLATFORM_NAME, 0, NULL, &valuesize);
     value = (char*) malloc (valuesize);
     clGetPlatformInfo (platform_id, CL_PLATFORM_NAME, valuesize, value, 0);
     fprintf(fp, "Platfrom: %s \n" , value);
     free(value);     

   //////////////////////////*** Print Device name


     clGetDeviceInfo (device_id, CL_DEVICE_NAME, 0, NULL, &valuesize);
     value = (char*) malloc (valuesize);
     clGetDeviceInfo (device_id, CL_DEVICE_NAME, valuesize, value, 0);
     fprintf(fp,"Device: %s \n",value);
     free(value);     

   ////////////////////////// Print Parallel compute units

     clGetDeviceInfo (device_id , CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits , NULL);
     fprintf(fp,"Parallel Compute Units : %d\n", maxComputeUnits);


     clGetDeviceInfo (device_id ,  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS , sizeof(workdimensions), &workdimensions , NULL);
     fprintf(fp,"Parallel Compute Units : %d\n", workdimensions);



  clGetDeviceInfo (device_id , CL_DEVICE_MAX_WORK_ITEM_SIZES , sizeof(maxWorkitem), maxWorkitem , NULL);


       fprintf(fp,"Maximum allowed work-item:");
     for(int i=0 ; i< 3 ; i++){
     fprintf(fp," %d", maxWorkitem[i]);
     }


clGetDeviceInfo (device_id , CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroup), &maxWorkGroup , NULL);     
     fprintf(fp,"\nMaximum allowed work group size: %d\n", maxWorkGroup);


    /////////// Device OpenCL version//
    //
      clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valuesize);
      value = (char *) malloc (valuesize);
      clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, valuesize, value, NULL);

      fprintf(fp, "OpenCL C version: %s \n" , value);

      free(value);


     //#Region 2: Context and Command Creation  

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS)
     print_error( "Failed to create context", __LINE__);


    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id,  0 , &err);
    if(err != CL_SUCCESS)
     print_error("Failed to create command queue", __LINE__);

         


    // #Region 3 : Buffer Creation and Input Write

 /////////////

     #ifdef PINNED
   
    cl_mem a_mem_pinned = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
    cl_mem b_mem_pinned = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR,
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
    cl_mem c_mem_pinned = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
    cl_mem d_mem_pinned = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, 
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
     #else
   // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);
    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            DATA_SIZE * sizeof(unsigned int), NULL, &ret);

    #endif

////////////
    unsigned  int *A = (unsigned int *)malloc(sizeof(unsigned int)*DATA_SIZE);
    unsigned  int *B = (unsigned int *)malloc(sizeof(unsigned int)*DATA_SIZE);
    unsigned  int *C = (unsigned int *)malloc(sizeof(unsigned int)*DATA_SIZE);
    unsigned  int *D = (unsigned int *)malloc(sizeof(unsigned int)*DATA_SIZE);

 /////////////////////////////////////
 //
    #ifdef PINNED
     A= (unsigned int *) clEnqueueMapBuffer(command_queue,a_mem_pinned, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, DATA_SIZE * sizeof(int),0, NULL, NULL, NULL);
     B= (unsigned int *) clEnqueueMapBuffer(command_queue,b_mem_pinned, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, DATA_SIZE * sizeof(int),0, NULL, NULL, NULL);
     C= (unsigned int *) clEnqueueMapBuffer(command_queue,c_mem_pinned, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE,  0, DATA_SIZE * sizeof(int) ,0, NULL, NULL, NULL);
     D= (unsigned int *) clEnqueueMapBuffer(command_queue,d_mem_pinned, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE,  0, DATA_SIZE * sizeof(int) ,0, NULL, NULL, NULL);
   
#endif
     
    ///////////////////
  srand(time(NULL));   

    for(i = 0; i < DATA_SIZE; i++) {

        B[i] = i;
         
           // Initialization, should only be called once.
       random = rand() % 1000 + 1 ; // Returns a pseudo-random integer between 0 and RAND_MAX.

        A[i]= (unsigned int)random;    
    }
   
 
  #ifdef PINNED            	

     clEnqueueUnmapMemObject(command_queue, a_mem_pinned, A, 0, NULL, NULL);
     clEnqueueUnmapMemObject(command_queue, b_mem_pinned, B, 0, NULL, NULL);
    
  #else
  
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            DATA_SIZE * sizeof(unsigned int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            DATA_SIZE * sizeof(unsigned int), B, 0, NULL, NULL);
 #endif


    // #Region 4:  Source code loading and compilation
   
     load_file_to_memory(filename, &kernelsource);

      cl_program  program = clCreateProgramWithSource(context, 1, (const char **) &kernelsource, NULL, &err);
           if(err != CL_SUCCESS)
	    print_error("Program sourcing is Failed", __LINE__);


    // Build the program
    ret = clBuildProgram(program, 1, &device_id, &options, NULL, err);
     if(err != CL_SUCCESS)
      print_error("Program Compilation is Failed", __LINE__);
     

    // #Region 5-1 : Kernel Creation and Arguments assignments   



    // Create the OpenCL kernel
    cl_kernel sortlocal_kernel = clCreateKernel(program, "bitonicSortLocal1", &ret);
             if(sortlocal_kernel == NULL) 
       print_error("Creating KERNEL IS FAILED", __LINE__);
	     
       cl_kernel mergeGlobalKernel = clCreateKernel(program, "bitonicMergeGlobal", &err);
          if(sortlocal_kernel == NULL)
       print_error("Creating KERNEL IS FAILED", __LINE__);

       cl_kernel mergeLocalKernel = clCreateKernel(program, "bitonicMergeLocal", &err); 
          if(sortlocal_kernel == NULL)
       print_error("Creating KERNEL IS FAILED", __LINE__);
/////////
        
	   
  ret=clGetKernelWorkGroupInfo (sortlocal_kernel,device_id,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE ,sizeof( preferred_groupsize), &preferred_groupsize,NULL);
     if(ret != CL_SUCCESS)
  print_error("GetkernelWorkGroup info is failed", __LINE__);
    fprintf(fp,"The preferred group size is:%d\n ",  preferred_groupsize);


    // Set the arguments of the kernel
    //
    #ifdef PINNED
    ret = clSetKernelArg(sortlocal_kernel, 0, sizeof(cl_mem), &c_mem_pinned);
    ret = clSetKernelArg(sortlocal_kernel, 1, sizeof(cl_mem), &d_mem_pinned);
    ret = clSetKernelArg(sortlocal_kernel, 2, sizeof(cl_mem), &a_mem_pinned);
    ret = clSetKernelArg(sortlocal_kernel, 3, sizeof(cl_mem), &b_mem_pinned);
    #else
    ret = clSetKernelArg(sortlocal_kernel, 0, sizeof(cl_mem), &c_mem_obj);
    ret = clSetKernelArg(sortlocal_kernel, 1, sizeof(cl_mem), &d_mem_obj);
    ret = clSetKernelArg(sortlocal_kernel, 2, sizeof(cl_mem), &a_mem_obj);
    ret = clSetKernelArg(sortlocal_kernel, 3, sizeof(cl_mem), &b_mem_obj);
    #endif
   

      
 /////// #Region 6-1 : Kernel Execution //////
 //
 //
           // te the OpenCL kernel on the list
   	 size_t global  = DATA_SIZE; // Process the entire lists
         size_t local   = LOCAL_SIZE / 2; 	

       ret = clEnqueueNDRangeKernel(command_queue, sortlocal_kernel, 1, NULL, 
            &global, &local, 0, NULL, NULL );
           if(ret!=CL_SUCCESS) 
           print_error("EXECUTION IS FAILED", __LINE__);
	   //////////////////////////////
	   //
	   //	   	   
////////////////////////////////////////////////////////

unsigned int total = 0;
        for (unsigned int size = 2 * local; size <= DATA_SIZE; size <<= 1) {
            for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
		total++;
	    }
	}
	unsigned int run = 0;
        for (unsigned int size = 2 * local; size <= DATA_SIZE; size <<= 1) {
            for (unsigned stride = size / 2; stride > 0; stride >>= 1) {
		run++;
             if(stride >= local) {
      #ifdef PINNED 
                 err |= clSetKernelArg(mergeGlobalKernel, 0, sizeof(cl_mem), &c_mem_pinned);
                 err |= clSetKernelArg(mergeGlobalKernel, 1, sizeof(cl_mem), &d_mem_pinned);
                 err |= clSetKernelArg(mergeGlobalKernel, 2, sizeof(cl_mem), &c_mem_pinned);
                 err |= clSetKernelArg(mergeGlobalKernel, 3, sizeof(cl_mem), &d_mem_pinned);
                 err |= clSetKernelArg(mergeGlobalKernel, 4, sizeof(cl_uint), &DATA_SIZE);
                 err |= clSetKernelArg(mergeGlobalKernel, 5, sizeof(cl_uint), &size);
                 err |= clSetKernelArg(mergeGlobalKernel, 6, sizeof(cl_uint), &stride);
                 err |= clSetKernelArg(mergeGlobalKernel, 7, sizeof(cl_uint), &dir);
       #else   
                 err |= clSetKernelArg(mergeGlobalKernel, 0, sizeof(cl_mem), &c_mem_obj);
                 err |= clSetKernelArg(mergeGlobalKernel, 1, sizeof(cl_mem), &d_mem_obj);
                 err |= clSetKernelArg(mergeGlobalKernel, 2, sizeof(cl_mem), &c_mem_obj);
                 err |= clSetKernelArg(mergeGlobalKernel, 3, sizeof(cl_mem), &d_mem_obj);
                 err |= clSetKernelArg(mergeGlobalKernel, 4, sizeof(cl_uint), &DATA_SIZE);
                 err |= clSetKernelArg(mergeGlobalKernel, 5, sizeof(cl_uint), &size);
                 err |= clSetKernelArg(mergeGlobalKernel, 6, sizeof(cl_uint), &stride);
                 err |= clSetKernelArg(mergeGlobalKernel, 7, sizeof(cl_uint), &dir);
       #endif

            //        printf("starting kernel MergeGlobal %2d out of %d (size %4u stride %4u)\n", run, total, size, stride); 

                   
                    err |= clEnqueueNDRangeKernel(command_queue, mergeGlobalKernel, 1, NULL, (size_t *)&global,(size_t *) &local, 0, NULL, NULL);
                  
                } else {

	         #ifdef PINNED		
                err |= clSetKernelArg(mergeLocalKernel, 0, sizeof(cl_mem), &c_mem_pinned);
                err |= clSetKernelArg(mergeLocalKernel, 1, sizeof (cl_mem),&d_mem_pinned);
                err |= clSetKernelArg(mergeLocalKernel, 2, sizeof(cl_mem), &c_mem_pinned); 
                err |= clSetKernelArg(mergeLocalKernel, 3, sizeof(cl_mem), &d_mem_pinned); 
                err |= clSetKernelArg(mergeLocalKernel, 4, sizeof(cl_uint), &DATA_SIZE); 
                err |= clSetKernelArg(mergeLocalKernel, 5, sizeof(cl_uint), &stride); 
                err |= clSetKernelArg(mergeLocalKernel, 6, sizeof(cl_uint), &size); 
                err |= clSetKernelArg(mergeLocalKernel, 7, sizeof(cl_uint), &dir); 
                 #else
	        err |= clSetKernelArg(mergeLocalKernel, 0, sizeof(cl_mem), &c_mem_obj);
                err |= clSetKernelArg(mergeLocalKernel, 1, sizeof (cl_mem),&d_mem_obj);
                err |= clSetKernelArg(mergeLocalKernel, 2, sizeof(cl_mem), &c_mem_obj); 
                err |= clSetKernelArg(mergeLocalKernel, 3, sizeof(cl_mem), &d_mem_obj); 
                err |= clSetKernelArg(mergeLocalKernel, 4, sizeof(cl_uint), &DATA_SIZE); 
                err |= clSetKernelArg(mergeLocalKernel, 5, sizeof(cl_uint), &stride); 
                err |= clSetKernelArg(mergeLocalKernel, 6, sizeof(cl_uint), &size); 
                err |= clSetKernelArg(mergeLocalKernel, 7, sizeof(cl_uint), &dir); 
                 #endif	
                if(err!=CL_SUCCESS)
           print_error("MergeGlobal is failed", __LINE__);

          //          printf("starting kernel MergeLocal  %2d out of %d (size %4u stride %4u)\n", run, total, size, stride); 
                    err |= clEnqueueNDRangeKernel(command_queue, mergeLocalKernel, 1, NULL, (size_t *)&global,(size_t *) &local, 0, NULL, NULL); 
                   if(err!=CL_SUCCESS)
           print_error("MergeLocal is failed", __LINE__);
            

                }
		err |= clFinish(command_queue);
            }
}	

    
////////// #Region 7 : Reading BAck kernel Buffers to Output
         
        
    // Read the memory buffer C on the device to the local variable C
      
      #ifdef PINNED
    
     clEnqueueUnmapMemObject(command_queue, c_mem_pinned, C, 0, NULL, NULL);
     clEnqueueUnmapMemObject(command_queue, d_mem_pinned, D, 0, NULL, NULL);

      #else 
     ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            DATA_SIZE * sizeof(unsigned int), C, 0, NULL, NULL);
     ret |= clEnqueueReadBuffer(command_queue, d_mem_obj, CL_TRUE, 0, 
            DATA_SIZE * sizeof(unsigned int), D, 0, NULL, NULL);
      if(ret != CL_SUCCESS)
           print_error("Unable to Read Back Buffers", __LINE__);
        


      #endif
    



     // #Region 8: Printing the results
  
     // Store inout and outputs in output.txt file
    
       #ifdef DEBUG
            fprintf(fp,"\n Computation Results:\n");

           for(i = 0; i < DATA_SIZE; i++)	{    
          	
              fprintf(fp, "%5d , %5d ,  %5d  \n", B[i], A[i], C[i]);
                }
  
             printf(BLUE "Open output.txt for further information \n"RESET);         
                   
             #endif

            fclose(fp);


      #ifdef PROFILER
        kernel_time= time_profiler(myevent, ret);
        printf ("Kernel execution time : %.5f \n", kernel_time);  
      #endif
      
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(sortlocal_kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    free(D);
    #ifndef PINNED
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseMemObject(d_mem_obj);
    #endif 

    return 0;

}





