// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSB_H
#define LBFGSB_H

#include <vector>
#include <stdexcept>  // std::invalid_argument
#include <Eigen/Core>
#include "LBFGS/Param.h"
#include "LBFGS/BFGSMat.h"
#include "LBFGS/Cauchy.h"
#include "LBFGS/LineSearchBacktracking.h"
#include "LBFGS/LineSearchBracketing.h"
#include "LBFGS/LineSearchNocedalWright.h"


namespace LBFGSpp {


///
/// LBFGSB solver for box-constrained numerical optimization
///
template < typename Scalar,
           template<class> class LineSearch = LineSearchBacktracking >
class LBFGSBSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector> MapVec;
    typedef std::pair<int, Scalar> BreakPoint;

    const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
    BFGSMat<Scalar>           m_bfgs;   // Approximation to the Hessian matrix
    Vector                    m_fx;     // History of the objective function values
    Vector                    m_xp;     // Old x
    Vector                    m_grad;   // New gradient
    Vector                    m_gradp;  // Old gradient
    Vector                    m_drt;    // Moving direction

    // Reset internal variables
    // n: dimension of the vector to be optimized
    inline void reset(int n)
    {
        const int m = m_param.m;
        m_bfgs.reset(n, m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
        if(m_param.past > 0)
            m_fx.resize(m_param.past);
    }

public:
    ///
    /// Constructor for LBFGS solver.
    ///
    /// \param param An object of \ref LBFGSParam to store parameters for the
    ///        algorithm
    ///
    LBFGSBSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    ///
    /// Minimizing a multivariate function using LBFGS algorithm.
    /// Exceptions will be thrown if error occurs.
    ///
    /// \param f  A function object such that `f(x, grad)` returns the
    ///           objective function value at `x`, and overwrites `grad` with
    ///           the gradient.
    /// \param x  In: An initial guess of the optimal point. Out: The best point
    ///           found.
    /// \param fx Out: The objective function value at `x`.
    ///
    /// \return Number of iterations used.
    ///
    template <typename Foo>
    inline int minimize(Foo& f, Vector& x, Scalar& fx, const Vector& lb, const Vector& ub)
    {
        // Dimension of the vector
        const int n = x.size();
        if(lb.size() != n || ub.size() != n)
            throw std::invalid_argument("'lb' and 'ub' must have the same size as 'x'");

        // Check whether the intiial vector is within the bounds
        if(!Cauchy<Scalar>::in_bounds(x, lb, ub))
            throw std::invalid_argument("initial 'x' is out of the bounds");

        // Initialization
        reset(n);

        // The length of lag for objective function value to test convergence
        const int fpast = m_param.past;

        // Evaluate function and compute gradient
        fx = f(x, m_grad);
        Scalar xnorm = x.norm();
        Scalar gnorm = m_grad.norm();
        if(fpast > 0)
            m_fx[0] = fx;

        Vector xcp;
        m_bfgs.form_M();
        Cauchy<Scalar>::get_cauchy_point(m_bfgs, x, m_grad, lb, ub, xcp);

        // Early exit if the initial x is already a minimizer
        if(gnorm <= m_param.epsilon * std::max(xnorm, Scalar(1.0)))
        {
            return 1;
        }

        // Initial direction
        m_drt.noalias() = -m_grad;
        // Initial step size
        Scalar step = Scalar(1.0) / m_drt.norm();

        // Number of iterations used
        int k = 1;
        for( ; ; )
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;

            // Line search to update x, fx and gradient
            LineSearch<Scalar>::LineSearch(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

            // New x norm and gradient norm
            xnorm = x.norm();
            gnorm = m_grad.norm();

            // Convergence test -- gradient
            if(gnorm <= m_param.epsilon * std::max(xnorm, Scalar(1.0)))
            {
                return k;
            }
            // Convergence test -- objective function value
            if(fpast > 0)
            {
                if(k >= fpast && std::abs((m_fx[k % fpast] - fx) / fx) < m_param.delta)
                    return k;

                m_fx[k % fpast] = fx;
            }
            // Maximum number of iterations
            if(m_param.max_iterations != 0 && k >= m_param.max_iterations)
            {
                return k;
            }

            // Update s and y
            // s_{k+1} = x_{k+1} - x_k
            // y_{k+1} = g_{k+1} - g_k
            m_bfgs.add_correction(x - m_xp, m_grad - m_gradp);

            // m_bfgs.form_M();

            // Recursive formula to compute d = -H * g
            m_bfgs.apply_Hv(m_grad, -Scalar(1), m_drt);

            // Reset step = 1.0 as initial guess for the next line search
            step = Scalar(1);
            k++;
        }

        return k;
    }
};


} // namespace LBFGSpp

#endif // LBFGSB_H
