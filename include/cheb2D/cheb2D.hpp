#pragma once

#include <iostream>
#include <memory>

#include <Eigen/Dense>

template < typename MatrixType>
Eigen::Matrix<double, MatrixType::RowsAtCompileTime, 1> Clenshawbycol(double yscaled, const MatrixType &m_c, Eigen::Index actualrows = -1){
    // See https ://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
    auto Norder = m_c.cols() - 1;
    Eigen::Array<double, MatrixType::RowsAtCompileTime, 1> u_k, u_kp1 = m_c.col(Norder), u_kp2;
    if constexpr(MatrixType::RowsAtCompileTime < 0){
        u_k.resize(actualrows);
        u_kp2.resize(actualrows);
    }
    u_k.setZero(); u_kp2.setZero();
    for (auto k = Norder - 1; k > 0; --k) {
        // Do the recurrent calculation
        u_k = 2.0*yscaled*u_kp1 - u_kp2 + m_c.col(k);
        // Update the values
        u_kp2 = u_kp1; u_kp1 = u_k;
    }
    return m_c.col(0) + yscaled*u_kp1 - u_kp2;
}

/* Classic Clenshaw method, templated */
template <typename VecType>
double Clenshaw(double xscaled, const VecType & m_c){
    // See https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series
    auto Norder = m_c.size() - 1;
    auto u_k = 0.0, u_kp1 = m_c[Norder], u_kp2 = 0.0;
    auto k = 0;
    for (auto k = Norder-1; k > 0; --k){
        // Do the recurrent calculation
        u_k = 2.0 * xscaled * u_kp1 - u_kp2 + m_c[k];
        // Update the values
        u_kp2 = u_kp1; u_kp1 = u_k;
    }
    return m_c[0] + xscaled * u_kp1 - u_kp2;
}

template <typename MatType>
Eigen::MatrixXd get_DCT_matrix(Eigen::Index N){
    Eigen::MatrixXd L(N+1,N+1); L.setZero(); // Matrix of coefficients
    for (auto j = 0; j <= N; ++j){
        for (auto k = j; k <= N; ++k){
            auto p_j = (j == 0 || j == N) ? 2 : 1;
            auto p_k = (k == 0 || k == N) ? 2 : 1;
            L(j, k) = 2.0 / (p_j * p_k * N) * cos((j * M_PI * k) / N);
            // Exploit symmetry to fill in the symmetric elements in the matrix
            L(k, j) = L(j, k);
        }
    }
    return L;
}

struct Cheb2DDomain{
    double xmin, xmax, ymin, ymax;
};

const double M_PI = acos(-1.0);

typedef std::function<double(double, double)> Func2D;
template <typename CoeffMatType, int Rows = CoeffMatType::RowsAtCompileTime, int Cols = CoeffMatType::ColsAtCompileTime>
class Cheb2D{
private:
    CoeffMatType coeff;
    Eigen::Index Nx, Ny;
    Cheb2DDomain spec;
    bool _resize_required;
    Eigen::Matrix<double, Rows, Rows> Lx;
    Eigen::Matrix<double, Cols, Cols> Ly;
public:
    Cheb2D(const Cheb2DDomain& spec) : spec(spec), _resize_required(false) {
        _resize_required = false;
        Nx = coeff.RowsAtCompileTime;
        Ny = coeff.ColsAtCompileTime;
        // Build the DCT matrices if the sizes are fixed
        if constexpr (Rows > 0){
            Lx = get_DCT_matrix<CoeffMatType>(Nx-1);
        }
        else{
            _resize_required = true;
        }
        if constexpr (Cols > 0){
            Ly = get_DCT_matrix<CoeffMatType>(Ny-1);
        }
        else{
            _resize_required = true;
        }
    };
    void resize(Eigen::Index rows, Eigen::Index cols){
        coeff.resize(rows, cols);
        if constexpr (Rows < 0) {
            Lx = get_DCT_matrix<CoeffMatType>(rows - 1);
        }
        if constexpr (Cols < 0) {
            Ly = get_DCT_matrix<CoeffMatType>(cols - 1);
        }
        _resize_required = false;
    }
    double scalex(double x) const {
        // Scale input x value from real-world into [-1,1]
        return (2*x - (spec.xmax + spec.xmin)) / (spec.xmax - spec.xmin);
    }
    double unscalex(double xn11) const {
        // Scale input x value in [-1,1] into real-world values
        return (xn11*(spec.xmax - spec.xmin) + (spec.xmax + spec.xmin))/2;
    }
    double scaley(double y) const {
        // Scale input y value from real-world into [-1,1]
        return (2 * y - (spec.ymax + spec.ymin)) / (spec.ymax - spec.ymin);
    }
    double unscaley(double yn11) const {
        // Scale input y value in [-1,1] into real-world values
        return (yn11 * (spec.ymax - spec.ymin) + (spec.ymax + spec.ymin))/2;
    }
    void build(const Func2D&f) {
        this->spec = spec;
        if (_resize_required){
            throw std::invalid_argument("Need to call resize since dimensions are dynamic");
        }
        
        using ColArray = Eigen::Array<double, Rows, 1>;
        ColArray ynodes_n11;
        ColArray j;
        if constexpr(Rows > 0){ // compile-time sized
            ynodes_n11 = cos(ColArray::LinSpaced(Rows, 0, Rows - 1).cast<double>()*M_PI/(Rows - 1));
        }
        else{
            int introws = static_cast<int>(coeff.rows());
            double actualrows = static_cast<double>(introws);
            ynodes_n11 = cos(ColArray::LinSpaced(introws, 0, actualrows - 1).cast<double>() * M_PI / (actualrows - 1));
        }

        using RowArray = Eigen::Array<double, Cols, 1>;
        RowArray xnodes_n11;
        if constexpr (Cols > 0) { // compile-time sized
            xnodes_n11 = cos(ColArray::LinSpaced(Cols, 0, Cols - 1).cast<double>() * M_PI / (Cols - 1));
        }
        else {
            int intcols = static_cast<int>(coeff.cols());
            double actualcols = static_cast<double>(intcols);
            xnodes_n11 = cos(ColArray::LinSpaced(intcols, 0, actualcols - 1).cast<double>() * M_PI / (actualcols - 1));
        }
        
        for (auto ir = 0; ir < ynodes_n11.size(); ++ir){
            auto y = unscaley(ynodes_n11(ir));

            Eigen::Matrix<double, Rows, 1> fnodes;
            if constexpr (Rows < 1){ fnodes.resize( coeff.cols() ); };

            // Function values at the x nodes along the y node
            for (auto j = 0; j < fnodes.size(); ++j){
                auto x = unscalex(xnodes_n11(j));
                fnodes(j) = f(x, y);
            }
            coeff.row(ir) = Lx*fnodes;
        }
    }
    double eval(double x, double y) const{

        // Flatten in x direction with Clenshaw for each row in C to get the functional
        // values of the expansion at the ynodes
        auto fatynodes = Clenshawbycol(scalex(x), coeff, coeff.rows()); // f at Chebshev - Lobatto ynodes for specified value of xscaled

        // Build expansion from functional values at y nodes
        auto c = Ly*fatynodes;
        return Clenshaw(scaley(y), c);
    }
};

struct Cheb2DSpec{   
    int Nx, Ny;
    Cheb2DDomain domain;
};

// Either this node is a terminal node in the tree or
// it has a reference to two further nodes
class Cheb2DNode{
private:
    bool terminal;
    std::shared_ptr<Cheb2DNode> child1, child2;
    Cheb2DSpec spec;
    double xmid = -1, ymid = -1;
    bool split_in_y = false, split_in_x = false;
    const Func2D f;

    using MatType = Eigen::Array<double, 8, 8>;
    Cheb2D<MatType> ce;
    Cheb2D<MatType> build(const Func2D& f, const Cheb2DSpec& spec){
        Cheb2D<MatType> ce(spec.domain);
        ce.build(f);
        return ce;
    }
public:
    Cheb2DNode(const Func2D &f, const Cheb2DSpec& spec) : f(f), spec(spec), ce(build(f, spec)){
        // I'm terminal
        terminal = true;
    }
    void dyadic_split_y(){
        if (!terminal) { throw std::invalid_argument("split can only be called on terminal node;"); }
        ymid = (spec.domain.ymin + spec.domain.ymax)/2;
        auto newspec1 = spec, newspec2 = spec;
        newspec1.domain.ymax = ymid;
        newspec2.domain.ymin = ymid;
        child1.reset(new Cheb2DNode(f, newspec1));
        child2.reset(new Cheb2DNode(f, newspec2));
        split_in_y = true;
        terminal = false;
    };
    void dyadic_split_x(){
        if (!terminal){ throw std::invalid_argument("split can only be called on terminal node;"); }
        xmid = (spec.domain.xmin + spec.domain.xmax) / 2;
        auto newspec1 = spec, newspec2 = spec;
        newspec1.domain.xmax = xmid;
        newspec2.domain.xmin = xmid;
        child1.reset(new Cheb2DNode(f, newspec1));
        child2.reset(new Cheb2DNode(f, newspec2));
        split_in_x = true;
        terminal = false;
    };
    Cheb2DNode& recursive_getnode(double x, double y) {
        // Handle the trivial case
        if (terminal) {
            return *this;
        }
        else if (split_in_x){
            // Recurse
            return (x >= xmid) ? child2->recursive_getnode(x,y) : child1->recursive_getnode(x,y);
        }
        else if (split_in_y) {
            // Recurse
            return (y >= ymid) ? child2->recursive_getnode(x,y) : child1->recursive_getnode(x,y);
        }
    }
    void recursive_split(bool xdirection) {
        // Handle the trivial case
        if (terminal) {
            if (xdirection){
                dyadic_split_x();
            }
            else{
                dyadic_split_y();
            }
        }
        else {
            child1->recursive_split(xdirection);
            child2->recursive_split(xdirection);
        }
    }
    bool in_bbox(double x, double y){
        return x >= spec.domain.xmin && x <= spec.domain.xmax && y >= spec.domain.ymin && y <= spec.domain.ymax;
    }
    auto & get_cheb2d(){
        return ce;
    }
};

class Cheb2DTree{
private:
    Cheb2DSpec spec;
    Cheb2DNode rootnode;
    Cheb2DNode get_rootnode(const Func2D& f){
        return Cheb2DNode(f, spec);
    }
public:
    Cheb2DTree(const Func2D & f, const Cheb2DSpec &spec) : spec(spec), rootnode(get_rootnode(f)) { };
    void recursive_split(bool xdirection){
        rootnode.recursive_split(xdirection);
    }
    Cheb2DNode& getnode(double x, double y) {
        return rootnode.recursive_getnode(x,y);
    }
    double eval(double x, double y){
        return rootnode.recursive_getnode(x,y).get_cheb2d().eval(x, y);
    }
};