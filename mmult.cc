#include <bits/stdint-intn.h>
#include <cstdio>
#include <celerity.h>
#include <cstdlib>
#include <hipSYCL/sycl/libkernel/range.hpp>
#include <random>

#define array(matrix,n,i,j) matrix[(int)n*(int)i+(int)j]

int MAT_SIZE = 2048;

void fill_random(float *A, const int &n)
{
  std::mt19937 e(static_cast<unsigned int>(time(nullptr)));
  std::uniform_real_distribution<float> f;
  for(int i=0; i<n; ++i)
  {
    for(int j=0; j<n; ++j)
    {
      array(A,n,i,j) = f(e);
    }
  }
}

int main(int argc, char* argv[]) {
	if(argc > 1){
		MAT_SIZE = atoi(argv[1]);
	}
	int i,j,k;
    int err = 0;
	float *array_a, *array_b, *array_c;
	// Array initiliazation at host
	array_a = new float[MAT_SIZE*MAT_SIZE];
	array_b = new float[MAT_SIZE*MAT_SIZE];
	array_c = new float[MAT_SIZE*MAT_SIZE];
    //arrays a and b are filled with random float values
    fill_random(array_a, MAT_SIZE);
    fill_random(array_b, MAT_SIZE);
	for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
			array(array_c,MAT_SIZE,i,j) = 0.0;
		}
	}
    auto range = cl::sycl::range<2>(MAT_SIZE, MAT_SIZE);
    celerity::buffer<float, 2> mat_c(range);
    celerity::buffer<float, 2> mat_a(array_a, cl::sycl::range<2>(MAT_SIZE,MAT_SIZE));
    celerity::buffer<float, 2> mat_b(array_b, cl::sycl::range<2>(MAT_SIZE,MAT_SIZE));
    celerity::distr_queue queue;
    queue.submit([=](celerity::handler& cgh) {
		auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
		auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
        cgh.parallel_for<class MmultKernel>(
            cl::sycl::range<2>(MAT_SIZE, MAT_SIZE),
            [=](cl::sycl::item<2> item) {
                float sum = 0.0;
                for(size_t k = 0; k < MAT_SIZE; ++k) {
                    float a_ik = a[{item[0], k}];
                    float b_kj = b[{k, item[1]}];
                    sum += a_ik * b_kj;
                }
                c[item] = sum;
            }
        );
    });
    queue.submit(celerity::allow_by_ref, [&](celerity::handler& cgh) {
        auto result = mat_c.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
        cgh.host_task(celerity::on_master_node, [=, &err, &array_c]() {
            const auto max_idx = MAT_SIZE;
            for(size_t i = 0; i < max_idx; ++i) {
                for(size_t j = 0; j < max_idx; ++j) {
                    for( size_t k =0; k < max_idx; k++ ){
                        //calculate the result of each element of the array at host
                        array(array_c,MAT_SIZE,i,j) = array(array_c,MAT_SIZE,i,j)+array(array_a,MAT_SIZE,i,k)*array(array_b,MAT_SIZE,k,j);
                    }
                    //compare the values calculated by host and the distributed tasks
                    auto host_res = array(array_c,MAT_SIZE,i,j);
                    auto kern_res = result[{i,j}];
                    if ( kern_res != host_res) {
                        fprintf(stderr,"Value Error at i= %zu, j= %zu value from host= %f, value from kernel= %f\n",i,j,host_res,kern_res);
                        err = 1;
                    }
                }
            }
        });
    });
    if(!err){
        printf("Succesful Multiplication\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("Unsuccesful Multiplication\n");
        return EXIT_FAILURE;
    }
}
