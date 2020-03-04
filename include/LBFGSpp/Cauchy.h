// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef CAUCHY_H
#define CAUCHY_H

#include <vector>
#include <Eigen/Core>
#include "BFGSMat.h"


/// \cond

namespace LBFGSpp {


//
// Class to compute the generalized Cauchy point (GCP) for the L-BFGS-B algorithm,
// mainly for internal use.
//
// The target of the GCP procedure is to find a step size t such that
// x(t) = x0 - t * g is a local minimum of the quadratic function m(x),
// where m(x) is a local approximation to the objective function.
//
// First determine a sequence of break points t0=0, t1, t2, ..., tn.
// On each interval [t[i-1], t[i]], x is changing linearly.
// After passing a break point, one or more coordinates of x will be fixed at the bounds.
// We search the first local minimum of m(x) by examining the intervals [t[i-1], t[i]] sequentially.
//
// Reference:
// [1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
//
template <typename Scalar>
class ArgSort
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> IntVector;

    const Vector& values;

public:
    ArgSort(const Vector& value_vec) :
        values(value_vec)
    {}

    inline bool operator()(int key1, int key2) { return values[key1] < values[key2]; }
    inline void sort_key(IntVector& key_vec) const
    {
        std::sort(key_vec.data(), key_vec.data() + key_vec.size(), *this);
    }
};

template <typename Scalar>
class Cauchy
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<int, Eigen::Dynamic, 1> IntVector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    // Find the smallest index i such that brk[ord[i]] > t, assuming brk[ord] is already sorted.
    // If the return value equals n, then all values are <= t.
    static int search_greater(const Vector& brk, const IntVector& ord, const Scalar& t)
    {
        const int n = brk.size();
        int i;
        for(i = 0; i < n; i++)
        {
            if(brk[ord[i]] > t)
                break;
        }

        return i;
    }

public:
    // bfgs: An object that represents the BFGS approximation matrix.
    // x0:   Current parameter vector.
    // g:    Gradient at x0.
    // lb:   Lower bounds for x.
    // ub:   Upper bounds for x.
    // xcp:  The output generalized Cauchy point.
    static void get_cauchy_point(const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& g, const Vector& lb, const Vector& ub, Vector& xcp)
    {
        // std::cout << "========================= Entering GCP search =========================\n\n";

        const int n = x0.size();
        xcp.resize(n);
        xcp.noalias() = x0;

        // Construct break points
        Vector brk(n), vecd(n);
        IntVector ord = IntVector::LinSpaced(n, 0, n - 1);
        for(int i = 0; i < n; i++)
        {
            if(g[i] < Scalar(0))
                brk[i] = (x0[i] - ub[i]) / g[i];
            else if(g[i] > Scalar(0))
                brk[i] = (x0[i] - lb[i]) / g[i];
            else
                brk[i] = std::numeric_limits<Scalar>::infinity();

            vecd[i] = (brk[i] == Scalar(0)) ? Scalar(0) : -g[i];
        }

        // Sort indices of break points
        ArgSort<Scalar> sorting(brk);
        sorting.sort_key(ord);

        // Break points `brko := brk[ord]` are in increasing order
        // `ord` contains the coordinates that define the corresponding break points
        // brk[i] == 0 <=> The i-th coordinate is on the boundary
        if(brk[ord[n - 1]] <= Scalar(0))
        {
            /* std::cout << "** All coordinates at boundary **\n";
            std::cout << "\n========================= Leaving GCP search =========================\n\n"; */
            return;
        }

        // First interval: [il=0, iu=brko[b], where b is the smallest index such that brko[b] > il
        // The corresponding coordinate that defines this break point is ord[b]

        // p = W'd
        Vector vecp;
        bfgs.apply_Wtv(vecd, vecp);
        // c = 0
        Vector vecc = Vector::Zero(vecp.size());
        // f' = -d'd
        Scalar fp = -vecd.squaredNorm();
        // f'' = -theta * f' - p'Mp
        Vector cache;
        bfgs.apply_Mv(vecp, cache);  // cache = Mp
        Scalar fpp = -bfgs.theta() * fp - vecp.dot(cache);

        // Theoretical step size to move
        Scalar deltatmin = -fp / fpp;

        // Limit on the current interval
        Scalar il = Scalar(0);
        // We have excluded the case that max(brk) <= 0
        int b = search_greater(brk, ord, il);
        Scalar iu = brk[ord[b]];
        Scalar deltat = iu - il;

        int iter = 0;
        /* std::cout << "** Iter " << iter << " **\n";
        std::cout << "   fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;
        std::cout << "   il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl; */

        // If deltatmin >= deltat, move to the next interval
        while(deltatmin >= deltat)
        {
            // First check how many coordinates will be active when we move to the previous iu
            // b is the smallest number such that brko[b] == iu
            // Let bp be the largest number such that brko[bp] == iu
            // Then coordinates ord[b] to ord[bp] will be active
            int bp = search_greater(brk, ord, iu) - 1;

            // Update xcp and d on active coordinates
            // std::cout << "** [ ";
            for(int i = b; i <= bp; i++)
            {
                const int coordb = ord[i];
                xcp[coordb] = (vecd[coordb] > Scalar(0)) ? ub[coordb] : lb[coordb];
                vecd[coordb] = Scalar(0);
                // std::cout << coordb + 1 << " ";
            }
            // std::cout << "] become active **\n\n";

            // If bp == n - 1, then we have reached the boundary of every coordinate
            if(bp == n - 1)
            {
                iter++;
                // std::cout << "** All break points visited **" << iter << std::endl;

                b = bp + 1;
                deltatmin = iu - il;
                break;
            }

            // Update a number of quantities after some coordinates become active
            vecc.noalias() += deltat * vecp;
            fp += deltat * fpp;
            Vector newvecp = vecp;
            for(int i = b; i <= bp; i++)
            {
                const int coordb = ord[i];
                // zb = xcpb - x0b
                const Scalar zb = xcp[coordb] - x0[coordb];
                const Scalar gb = g[coordb];
                const Scalar ggb = gb * gb;
                const Vector wb = bfgs.Wb(coordb);
                bfgs.apply_Mv(wb, cache);  // cache = Mw
                fp += ggb + bfgs.theta() * gb * zb - gb * cache.dot(vecc);
                fpp += -(bfgs.theta() * ggb - 2 * gb * cache.dot(vecp) - ggb * cache.dot(wb));
                // p = p + gb * wb
                newvecp.noalias() += gb * wb;
                // db = 0
                vecd[coordb] = Scalar(0);
            }

            // Theoretical step size to move
            deltatmin = -fp / fpp;

            // Update interval bound
            il = iu;
            b = bp + 1;
            iu = brk[ord[b]];

            // Limit on the current interval
            deltat = iu - il;

            iter++;
            /* std::cout << "** Iter " << iter << " **\n";
            std::cout << "   fp = " << fp << ", fpp = " << fpp << ", deltatmin = " << deltatmin << std::endl;
            std::cout << "   il = " << il << ", iu = " << iu << ", deltat = " << deltat << std::endl; */
        }

        // Last step
        const Scalar tfinal = il + std::max(deltatmin, Scalar(0));
        for(int i = b; i < n; i++)
        {
            const int coordb = ord[i];
            xcp[coordb] = x0[coordb] + tfinal * vecd[coordb];
        }
        // std::cout << "\n========================= Leaving GCP search =========================\n\n";
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // CAUCHY_H
