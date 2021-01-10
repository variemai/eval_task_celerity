#include <cstddef>
#include <cstdio>
#include <random>
#include <cstdlib>
#include <celerity.h>
#include <hipSYCL/sycl/libkernel/range.hpp>

#define array(matrix,n,i,j) matrix[(int)n*(int)i+(int)j]

int MAT_SIZE = 4;

void fill_random(float *A, const int &n, const int &m)
{
  std::mt19937 e(static_cast<unsigned int>(time(nullptr)));
  std::uniform_real_distribution<float> f;
  for(int i=0; i<n; ++i)
  {
    for(int j=0; j<m; ++j)
    {
      array(A,n,i,j) = f(e);
    }
  }
}

void LU(float *A, float *L, float *U)
{
    int n = MAT_SIZE;
    for(int i=0;  i<n; i++){
        for(int j=0; j<n; j++){
            if(j>i){
                array(U,n,j,i) = 0.0f;
            }
            array(U,n,i,j) = array(A,n,i,j);
            for(int k=0; k<i; k++){
                array(U,n,i,j) -= array(U,n,k,j)*array(L,n,i,k);
            }
        }
        for(int j=0; j<n; j++){
            if(i>j){
                array(L,n,j,i) = 0.0f;
            }
            else if (j==i){
                array(L,n,j,i) = 1.0f;
            }
            else{
                array(L,n,j,i) = array(A,n,j,i) / array(U,n,i,i);
                for(int k=0; k<i; k++){
                    array(L,n,j,i) -= ((array(U,n,k,i) * array(L,n,j,k)) / array(U,n,i,i));
                }
            }
        }
    }

}

template <typename T>
void lu_d(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_u, celerity::buffer<T, 2> mat_l) {
	queue.submit([=](celerity::handler& cgh) {
		auto l = mat_l.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::all<2>());
		auto u = mat_u.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::all<2>());
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::all<2>());

        cgh.parallel_for<class lu_u>(cl::sycl::range<1>(MAT_SIZE),[=](cl::sycl::item<1> item){
            for (size_t j=0; j < MAT_SIZE; j++) {
                if(j > item[0]){
                    u[{j,item[0]}] = 0.0f;
                }
                auto s =  a[{item[0],j}];
                for(size_t k = 0; k < MAT_SIZE; ++k) {
                    auto u_kj = u[{k, j}];
                    auto l_ik = l[{item[0], k}];
                    s -= (u_kj * l_ik);
                }
                u[{item[0],j}] = s;
            }
            for (size_t j=0; j < MAT_SIZE; j++) {
                if(item[0] > j){
                    l[{j,item[0]}] = 0.0f;
                }
                else if ( item[0] == j ){
                    l[{j,item[0]}] = 1.0f;
                }
                else{
                    auto s = a[{j,item[0]}] / u[{item[0],item[0]}];
                    for(size_t k = 0; k < MAT_SIZE; ++k) {
                        auto u_ki = u[{k, item[0]}];
                        auto l_pk = l[{j, k}];
                        auto tmp = u_ki * l_pk;
                        s -= tmp / u[{item[0],item[0]}];
                    }
                    l[{j,item[0]}] = s;
                }
            }
        });
	});
}
int main(int argc, char* argv[]) {

	int rank;
	int i,j,k;
	float *A, *L, *U;

    //if user argument is not provided use the default size (4)
    if(argc > 1){
        MAT_SIZE = atoi(argv[1]);
    }
	// Array initiliazation at host
	A = new float[MAT_SIZE*MAT_SIZE];
	L = new float[MAT_SIZE*MAT_SIZE];
	U = new float[MAT_SIZE*MAT_SIZE];
	fill_random(A, MAT_SIZE, MAT_SIZE);
    for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
			array(L,MAT_SIZE,i,j) = 0.0f;
			array(U,MAT_SIZE,i,j) = 0.0f;
		}
	}
    // Run LU at host and get the results for verification
	LU(A,L,U);
    printf("Array A:\n");
	for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
			printf("%2.2f ",array(A,MAT_SIZE,i,j));
		}
		printf("\n");
	}
	printf("Array L:\n");
	for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
			printf("%2.2f ",array(L,MAT_SIZE,i,j));
		}
		printf("\n");
	}
    printf("Array U:\n");
	for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
			printf("%2.2f ",array(U,MAT_SIZE,i,j));
		}
		printf("\n");
	}

	for (i = 0; i < MAT_SIZE; i++) {
		for (j = 0; j < MAT_SIZE; j++) {
			array(L,MAT_SIZE,i,j) = 0.0f;
			array(U,MAT_SIZE,i,j) = 0.0f;
		}
	}

    celerity::distr_queue queue;
    celerity::buffer<float, 2> l(L, cl::sycl::range<2>(MAT_SIZE,MAT_SIZE));
    celerity::buffer<float, 2> u(U, cl::sycl::range<2>(MAT_SIZE,MAT_SIZE));
    celerity::buffer<float, 2> a(A, cl::sycl::range<2>(MAT_SIZE,MAT_SIZE));

    lu_d(queue, a, u, l); //kernel call

    queue.submit([=](celerity::handler& cgh) {
        auto uresult = u.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
        auto lresult = l.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::all<2>());
        cgh.host_task(celerity::on_master_node, [=] {

			printf("L After\n");
			for(size_t i = 0; i < MAT_SIZE; ++i) {
				for(size_t j = 0; j < MAT_SIZE; ++j) {
                    printf("%2.2f ",lresult[{i,j}]);
				}
                printf("\n");
			}
			printf("U After\n");
			for(size_t i = 0; i < MAT_SIZE; ++i) {
				for(size_t j = 0; j < MAT_SIZE; ++j) {
                    printf("%2.2f ",uresult[{i,j}]);
				}
                printf("\n");
			}

		});
	});

    return 0;
}
