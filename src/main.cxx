#include "cheb2D/cheb2D.hpp"
#include <chrono>

template <int Nx, int Ny>
void time_one(){
    using MatType = Eigen::Array<double, Nx, Ny>;
    Cheb2DDomain domain;
    domain.xmin = -1;
    domain.xmax = 1;
    domain.ymin = -1;
    domain.ymax = 1;
    Cheb2D<MatType> cheb2(domain);
    auto f = [](double x, double y) {return sin(x) * exp(-x*x -y*y); };
    if (Nx < 0) {
        cheb2.resize(Ny, Ny);
    }
    cheb2.build(f);

    int N = 1000*1000;
    volatile auto r = 0.0, x = 0.1, y = 0.7;
    auto startTime = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i) {
        r += cheb2.eval(x, y);
    }
    auto endTime = std::chrono::system_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N * 1e6;
    std::cout << Nx << "," << Ny << "," << Ny*Ny << "," << elap_us << " val: " <<  r/N << std::endl;

    //auto fcheb = cheb2.eval(0.7, 0.7);
    //auto ff = f(0.7, 0.7);
}

int main(){
    Cheb2DDomain domain;
    domain.xmin = -1;
    domain.xmax = 1;
    domain.ymin = -1;
    domain.ymax = 1;
    Cheb2DSpec spec;
    spec.domain = domain;
    auto f = [](double x, double y) {return sin(x) * exp(-x * x - y * y); };
    Cheb2DTree tree(f, spec);
    auto& node0 = tree.getnode(0.7, 0.7);
    bool xdirection = false;
    for (auto doublesplit = 0; doublesplit < 4; ++doublesplit){
        tree.recursive_split(xdirection);
        tree.recursive_split(!xdirection);
    }
    auto & node = tree.getnode(0.7, 0.7);
    {
        int N = 1000*1000;
        volatile auto r = 0.0, x = 0.1, y = 0.7;
        auto startTime = std::chrono::system_clock::now();
        for (int i = 0; i < N; ++i) {
            r += tree.eval(x, y);
        }
        auto endTime = std::chrono::system_clock::now();
        auto elap_us = std::chrono::duration<double>(endTime - startTime).count() / N * 1e6;
        std::cout << elap_us << " val:" << r/N/f(x, y)-1 << " rel. err." << std::endl;
    }

    time_one<2,2>();
    time_one<-1,2>();
    time_one<3,3>();
    time_one<-1,3>();
    time_one<4,4>(); 
    time_one<-1,4>();
    time_one<5,5>();
    time_one<-1,5>();
    time_one<6,6>();
    time_one<-1,6>();
    time_one<7,7>();
    time_one<-1,7>();
    time_one<8,8>();
    time_one<-1,8>();
    time_one<9,9>();
    time_one<-1, 9>();
    time_one<16,16>();
    int rr = 0;
}