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
#include "convolution.h"
#include "math.h"

size_t localWorkSize[2], globalWorkSize[2];

int main(int argc, char** argv) {

    // Create the two input vectors
    unsigned int i;
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
    unsigned int iteration=16;      
    char *kernelsource;
    char filename[]= "ConvolutionSeparable.cl";

 
// constants
    const unsigned int imageW = 3072;
    const unsigned int imageH = 3072; 

///    
    cl_mem a_in_mem , b_in_mem , c_out_mem , d_out_mem;   //OpenCL memory buffer objects  
//**  
     ////*** Host_memory  
     cl_float*  a_in =   (cl_float *) malloc(imageW * imageH * sizeof(cl_float));
     cl_float*  b_in=    (cl_float *) malloc(FULL_FILTER   * sizeof(cl_float)); 
     cl_float*  d_out =  (cl_float *) malloc(imageW * imageH * sizeof(cl_float));
     cl_float*  c_out_host =  (cl_float *) malloc(imageW * imageH * sizeof(cl_float));
     cl_float*  d_out_host =  (cl_float *) malloc(imageW * imageH * sizeof(cl_float));

//***
//
      
   


	//// Initializing host memory
	
       for(unsigned int i = 0; i < imageW * imageH; i++)
            a_in[i] = (cl_float)(rand() % 16);


      for(unsigned int i = 0; i < FULL_FILTER; i++)
            b_in[i] = (cl_float)(rand() % 16);

       
 //**** end of initializing hostmemory
 

////

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
     

//     fprintf(fp,"GLOBAL SIZE:%d , LOCAL SIZE:%d, Total WORK GROUP %d\n", DATA_SIZE, LOCAL_SIZE/2, DATA_SIZE/LOCAL_SIZE);

       

    ////// #Regione 1: Get the Device and Platfrom info and IDs
    //
    //
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
       if(ret!=CL_SUCCESS)
       print_error("Platform is not detected", __LINE__);
       
     ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
            &device_id, &ret_num_devices);      
       if(ret != CL_SUCCESS)
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
     fprintf(fp,"Maximum allowed work item in each direction : %d\n", workdimensions);



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
 //
 //
#ifdef PINNED
    
    printf("PINNED\n");
    a_in_mem = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR ,  imageW * imageH * sizeof(cl_float), a_in , &ret);
        
   b_in_mem = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, FULL_FILTER * sizeof(cl_float), b_in , &ret);
   	
   c_out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, imageW * imageH * sizeof(cl_float), NULL, &ret);
	
   d_out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, imageW * imageH * sizeof(cl_float), NULL, &ret);

#else  

  a_in_mem = clCreateBuffer(context, CL_MEM_READ_WRITE ,  imageW * imageH * sizeof(cl_float),NULL , &ret);
        
   b_in_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, FULL_FILTER * sizeof(cl_float), NULL , &ret);
   	
   c_out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, imageW * imageH * sizeof(cl_float), NULL, &ret);
	
   d_out_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, imageW * imageH * sizeof(cl_float), NULL, &ret);
  

   ret = clEnqueueWriteBuffer(command_queue, a_in_mem, CL_TRUE, 0, imageW * imageH * sizeof(cl_float), a_in,0, NULL, NULL);

   ret = clEnqueueWriteBuffer(command_queue, b_in_mem, CL_TRUE, 0,FULL_FILTER * sizeof(cl_float), b_in,0, NULL, NULL);


#endif

//**** End of Creating Buffer
//
    // #Region 4:  Source code loading and compilation
   
     load_file_to_memory(filename, &kernelsource);

      cl_program  program = clCreateProgramWithSource(context, 1, (const char **) &kernelsource, NULL, &err);
           if(err != CL_SUCCESS)
	    print_error("Program sourcing is Failed", __LINE__);


    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
     if(ret != CL_SUCCESS)
      print_error("Program Compilation is Failed", __LINE__);
        if(ret ==  CL_BUILD_PROGRAM_FAILURE){
	// Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("%s\n", log);
	}
     
     // #Region 5-1 : Kernel Creation and Arguments assignments   

    // Create the OpenCL kernel
       cl_kernel ConvolutionRows_kernel = clCreateKernel(program, "Horizontalconv", &err);
             if(err != CL_SUCCESS) 
       print_error("Creating KERNEL IS FAILED", __LINE__);
	     
   cl_kernel ConvolutionColumns_kernel = clCreateKernel(program, "Verticalconv", &err);
   
          if(err != CL_SUCCESS)
       print_error("Creating KERNEL IS FAILED", __LINE__);

////////////
        
	   
  ret=clGetKernelWorkGroupInfo (ConvolutionRows_kernel,device_id,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE ,sizeof( preferred_groupsize), &preferred_groupsize,NULL);
     if(ret != CL_SUCCESS)
  print_error("GetkernelWorkGroup info is failed", __LINE__);
    fprintf(fp,"The preferred group size is:%d\n ",  preferred_groupsize);
   
    
    

    // Set the arguments of the kernel
    //
    //
   
  ret|= clSetKernelArg(ConvolutionRows_kernel, 0, sizeof(cl_mem),  &c_out_mem);
  ret|= clSetKernelArg(ConvolutionRows_kernel, 1, sizeof(cl_mem),  &a_in_mem);
  ret|= clSetKernelArg(ConvolutionRows_kernel, 2, sizeof(cl_mem),  &b_in_mem);
  ret|= clSetKernelArg(ConvolutionRows_kernel, 3, sizeof(unsigned int), (void*)&imageW);
  ret|= clSetKernelArg(ConvolutionRows_kernel, 4, sizeof(unsigned int), (void*)&imageH);
  ret|= clSetKernelArg(ConvolutionRows_kernel, 5, sizeof(unsigned int), (void*)&imageW);
    
       
 /////// #Region 6-1 : Kernel Execution //////
 //
 //
           // te the OpenCL kernel on the list
  
    localWorkSize[0] = H_LOCAL_X;
    localWorkSize[1] = H_LOCAL_Y;
    globalWorkSize[0] = imageW / H_OUT_SIZE;
    globalWorkSize[1] = imageH;


       ret = clEnqueueNDRangeKernel(command_queue, ConvolutionRows_kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
           if(ret!=CL_SUCCESS) 
           print_error("EXECUTION IS FAILED", __LINE__);
	   //////////////////////////////
	   //
  ret|= clSetKernelArg(ConvolutionColumns_kernel ,0, sizeof(cl_mem), &d_out_mem);
  ret|= clSetKernelArg(ConvolutionColumns_kernel ,1, sizeof(cl_mem), &c_out_mem);
  ret|= clSetKernelArg(ConvolutionColumns_kernel ,2, sizeof(cl_mem), &b_in_mem);
  ret|= clSetKernelArg(ConvolutionColumns_kernel ,3, sizeof(unsigned int), (void*)&imageW);
  ret|= clSetKernelArg(ConvolutionColumns_kernel ,4, sizeof(unsigned int), (void*)&imageH);
  ret|= clSetKernelArg(ConvolutionColumns_kernel ,5, sizeof(unsigned int), (void*)&imageW);
  
    


    localWorkSize[0] = V_LOCAL_X;
    localWorkSize[1] = V_LOCAL_Y;
    globalWorkSize[0] = imageW;
    globalWorkSize[1] = imageH / V_OUT_SIZE;



    ret = clEnqueueNDRangeKernel(command_queue, ConvolutionColumns_kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
           if(ret!=CL_SUCCESS) 
           print_error("EXECUTION IS FAILED", __LINE__);






//////////////////////////////////////////////////////////////
   
////////// #Region 7 : Reading BAck kernel Buffers to Output
      
    // Read the memory buffer C on the device to the local variable C
     ret = clEnqueueReadBuffer(command_queue, d_out_mem , CL_TRUE, 0, 
         imageW * imageH * sizeof(cl_float), d_out, 0, NULL, NULL);
       if(ret!=CL_SUCCESS)
        print_error("Read Bufferr is failed", __LINE__);
     

       

     // #Region 8: Result Validation
       convolutionRowHost(c_out_host, a_in , b_in, imageW, imageH, HALF_FILTER );
       convolutionColumnHost(d_out_host, c_out_host , b_in, imageW, imageH, HALF_FILTER);
      double sum = 0, delta = 0;
        double L2norm;
      
/*      
*/
  
        err=0;
	for(unsigned int i = 0 ; i < imageW * imageH ; i++)
          
		
	{        
             if(d_out[i]-d_out_host[i] !=0 )
		
		{

      	 	err=1;

	    fprintf(fp,"ERROR: Host output :%f and Device output:%f and Pixel:%f \n", d_out_host[i], d_out[i], i);
        }
  }

	if(err==0)
	
	{
	fprintf(fp,"\n\n Test is Passed, performance metrics can be analyzed using SCOREP trace files. \n\n");	

	}


        printf("\n Please open output.txt file for more information\n");

       
       fclose(fp);



     
    // Clean up
    ret = clFlush(command_queue);
    ret = clReleaseKernel(ConvolutionRows_kernel);
    ret = clReleaseKernel(ConvolutionColumns_kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(a_in);
    free(b_in);
    free(d_out);
      return 0;

}





