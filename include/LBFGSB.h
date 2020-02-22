// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGSB_H
#define LBFGSB_H

#include <vector>
#include <stdexcept>  // std::invalid_argument
#include <Eigen/Core>
#include "LBFGS/Param.h"
#include "LBFGS/BFGSMat.h"
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

    // Check whether the vector is within the bounds
    static bool in_bounds(const Vector& x, const Vector& lb, const Vector& ub)
    {
        const int n = x.size();
        for(int i = 0; i < n; i++)
        {
            if(x[i] < lb[i] || x[i] > ub[i])
                return false;
        }
        return true;
    }

    // Project the vector x to the bound constraint set
    static void force_bounds(Vector& x, const Vector& lb, const Vector& ub)
    {
        x.noalias() = x.cwiseMax(lb).cwiseMin(ub);
    }

    static bool pair_comparison(const BreakPoint& t1, const BreakPoint& t2)
    {
        return t1.second < t2.second;
    }

    // Find the smallest index i such that brk[i].second > t, assuming brk.second is already sorted
    // If return value equals n, then all values are <= t
    static int search_greater(const std::vector<BreakPoint>& brk, const Scalar& t)
    {
        const int n = brk.size();
        int i;
        for(i = 0; i < n; i++)
        {
            if(brk[i].second > t)
                break;
        }

        return i;
    }

    // The target of the generalized Cauchy point (GCP) procedure is to find a step size `t` such that
    // x(t) = x0 - t * g is a local minimum of the quadratic function m(x)
    // First determine a sequence of break points t0=0, t1, t2, ..., tn
    // On each interval [t[i-1], t[i]], x is changing linearly
    // After passing a break point, one or more coordinates of x will be fixed at the bounds
    // We search the first local minimum of m(x) by examining the intervals [t[i-1], t[i]] sequentially
    inline void get_cauchy_point(const Vector& x0, const Vector& g, const Vector& lb, const Vector& ub, Vector& xcp)
    {
        const int n = x0.size();
        xcp.resize(n);
        xcp.noalias() = x0;

        // Construct break points
        std::vector<BreakPoint> brk(n);
        Vector vecd(n);
        for(int i = 0; i < n; i++)
        {
            if(g[i] < Scalar(0))
                brk[i] = std::make_pair(i, (x0[i] - ub[i]) / g[i]);
            else if(g[i] > Scalar(0))
                brk[i] = std::make_pair(i, (x0[i] - lb[i]) / g[i]);
            else
                brk[i] = std::make_pair(i, std::numeric_limits<Scalar>::infinity());

            vecd[i] = (brk[i].second == Scalar(0)) ? Scalar(0) : -g[i];
        }

        // Sort break points
        std::sort(brk.begin(), brk.end(), pair_comparison);

        // Notation: brk => brk.second, e.g. brk[i] => brk[i].second
        //           ord => brk.first, e.g. ord[i] => brk[i].first
        // Break points `brk` are in increasing order
        // `ord` contains the coordinates that define the corresponding break points
        // brk[i] == 0 <=> The ord[i]-th coordinate is on the boundary
        if(brk[n - 1].second <= Scalar(0))
            return;

        // First interval: [il=0, iu=brk[b]], where b is the smallest index such that brk[b] > il
        // The corresponding coordinate that defines this break point is ord[b]

        // p = W'd
        Vector vecp;
        m_bfgs.apply_Wtv(vecd, vecp);
        // c = 0
        Vector vecc = Vector::Zero(vecp.size());
        // f' = -d'd
        Scalar fp = -vecd.squaredNorm();
        // f'' = -theta * f' - p'Mp
        Vector cache;
        m_bfgs.apply_Mv(vecp, cache);  // cache = Mp
        Scalar fpp = -m_bfgs.theta() * fp - vecp.dot(cache);

        // Theoretical step size to move
        Scalar deltatmin = -fp / fpp;

        // Limit on the current interval
        Scalar il = Scalar(0);
        // We have excluded the case that max(brk) <= 0
        int b = search_greater(brk, il);
        Scalar iu = brk[b].second;
        Scalar deltat = iu - il;

        // If deltatmin >= deltat, move to the next interval
        while(deltatmin >= deltat)
        {
            // First check how many coordinates will be active when we move to the previous iu
            // b is the smallest number such that brk[b] == iu
            // Let bp be the largest number such that brk[bp] == iu
            // Then coordinates ord[b] to ord[bp] will be active
            int bp = search_greater(brk, iu) - 1;

            // Update xcp and d on active coordinates
            // std::cout << "[ ";
            for(int i = b; i <= bp; i++)
            {
                const int coordb = brk[i].first;
                xcp[coordb] = (vecd[coordb] > Scalar(0)) ? ub[coordb] : lb[coordb];
                vecd[coordb] = Scalar(0);
                std::cout << coordb + 1 << " ";
            }
            // std::cout << "]\n";
            // std::cout << xcp.transpose() << std::endl << std::endl;

            // If bp == n - 1, then we have reached the boundary of every coordinate
            if(bp == n - 1)
            {
                b = bp + 1;
                deltatmin = iu - il;
                break;
            }

            vecc.noalias() += deltat * vecp;
            fp += deltat * fpp;
            Vector newvecp = vecp;
            for(int i = b; i <= bp; i++)
            {
                const int coordb = brk[i].first;
                // zb = xcpb - x0b
                const Scalar zb = xcp[coordb] - x0[coordb];
                const Scalar gb = g[coordb];
                const Scalar ggb = gb * gb;
                const Vector wb = m_bfgs.wb(coordb);
                m_bfgs.apply_Mv(wb, cache);  // cache = Mw
                fp += ggb + m_bfgs.theta() * gb * zb - gb * cache.dot(vecc);
                fpp += -(m_bfgs.theta() * ggb - 2 * gb * cache.dot(vecp) - ggb * cache.dot(wb));
                // p = p + gb * wb
                newvecp.noalias() += gb * wb;
                // db = 0
                vecd[coordb] = Scalar(0);
            }

            // Theoretical step size to move
            deltatmin = -fp / fpp;
            // std::cout << "fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;

            // Update interval bound
            il = iu;
            b = bp + 1;
            iu = brk[b].second;

            // Limit on the current interval
            deltat = iu - il;
            // std::cout << "il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl;
        }

        const Scalar tfinal = il + std::max(deltatmin, Scalar(0));
        for(int i = b; i < n; i++)
        {
            const int coordb = brk[i].first;
            xcp[coordb] = x0[coordb] + tfinal * vecd[coordb];
        }
        // std::cout << std::endl << xcp.transpose() << std::endl;
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
        get_cauchy_point(x, m_grad, lb, ub, xcp);

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
