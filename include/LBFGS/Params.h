// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef PARAMS_H
#define PARAMS_H

#include <Eigen/Core>
#include <stdexcept>  // std::invalid_argument


namespace LBFGSpp {


template <typename Scalar = double>
class LBFGSParam
{
public:
    ///
    /// The number of corrections to approximate the inverse Hessian matrix.
    ///
    int    m;
    ///
    /// Tolerance for convergence test.
    /// Algorithm stops if \f$||g|| < \epsilon * \max(1, ||x||)\f$.
    ///
    Scalar epsilon;
    ///
    /// The maximum number of iterations.
    ///
    int    max_iterations;
    ///
    /// The maximum number of trials for the line search.
    ///
    int    max_linesearch;
    ///
    /// Parameter that controls the line search procedure.
    ///
    Scalar ftol;
    ///
    /// Coefficient for the Wolfe condition.
    ///
    Scalar wolfe;

public:
    LBFGSParam()
    {
        m              = 6;
        epsilon        = Scalar(1e-5);
        max_iterations = 0;
        max_linesearch = 20;
        ftol           = Scalar(1e-4);
        wolfe          = Scalar(0.9);
    }

    inline void check_param() const
    {
        if(m <= 0)
            throw std::invalid_argument("m must be positive");
        if(epsilon <= 0)
            throw std::invalid_argument("epsilon must be positive");
        if(max_iterations < 0)
            throw std::invalid_argument("max_iterations must be non-negative");
        if(max_linesearch <= 0)
            throw std::invalid_argument("max_linesearch must be positive");
        if(ftol <= 0 || ftol >= 0.5)
            throw std::invalid_argument("ftol must be 0 < ftol < 0.5");
        if(wolfe <= ftol || wolfe >= 1)
            throw std::invalid_argument("wolfe must be ftol < wolfe < 1");
    }
};


} // namespace LBFGSpp

#endif // PARAMS_H
