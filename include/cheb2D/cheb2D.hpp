#pragma once

#include <iostream>

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

struct Cheb2DSpec{
    double xmin, xmax, ymin, ymax;
};

const double M_PI = acos(-1.0);

template <typename CoeffMatType, int Rows = CoeffMatType::RowsAtCompileTime, int Cols = CoeffMatType::ColsAtCompileTime>
class Cheb2D{
private:
    CoeffMatType coeff;
    Eigen::Index Nx, Ny;
    Cheb2DSpec spec;
    bool _resize_required;
    Eigen::Matrix<double, Rows, Rows> Lx;
    Eigen::Matrix<double, Cols, Cols> Ly;
public:
    Cheb2D(const Cheb2DSpec& spec) : spec(spec), _resize_required(false) {
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
    void build(const std::function<double(double, double)> &f){
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
    double eval(double x, double y){

        // Flatten in x direction with Clenshaw for each row in C to get the functional
        // values of the expansion at the ynodes
        auto fatynodes = Clenshawbycol(scalex(x), coeff, coeff.rows()); // f at Chebshev - Lobatto ynodes for specified value of xscaled

        // Build expansion from functional values at y nodes
        auto c = Lx*fatynodes;
        return Clenshaw(scaley(y), c);
    }
};