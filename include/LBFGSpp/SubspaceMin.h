// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef SUBSPACE_MIN_H
#define SUBSPACE_MIN_H

#include <stdexcept>
#include <vector>
#include <Eigen/Core>
#include "BFGSMat.h"


/// \cond

namespace LBFGSpp {


//
// Subspace minimization procedure of the L-BFGS-B algorithm,
// mainly for internal use.
//
// The target of subspace minimization is to minimize the quadratic function m(x)
// over the free variables, subject to the bound condition.
// Free variables stand for coordinates that are not at the boundary in xcp,
// the generalized Cauchy point.
//
// In the classical implementation of L-BFGS-B [1], the minimization is done by first
// ignoring the box constraints, followed by a line search. Our implementation is
// an exact minimization subject to the bounds, based on the BOXCQP algorithm [2].
//
// Reference:
// [1] R. H. Byrd, P. Lu, and J. Nocedal (1995). A limited memory algorithm for bound constrained optimization.
// [2] C. Voglis and I. E. Lagaris (2004). BOXCQP: An algorithm for bound constrained convex quadratic problems.
//
template <typename Scalar>
class SubspaceMin
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef std::vector<int> IndexSet;

    // Correct out-of-bound values, and record the indices of active set and free variables
    static void analyze_boundary(Vector& x, const Vector& lb, const Vector& ub, IndexSet& act_ind, IndexSet& fv_ind)
    {
        const int n = x.size();
        act_ind.clear();
        fv_ind.clear();

        for(int i = 0; i < n; i++)
        {
            if(x[i] >= ub[i])
            {
                x[i] = ub[i];
                act_ind.push_back(i);
            } else if(x[i] <= lb[i]) {
                x[i] = lb[i];
                act_ind.push_back(i);
            } else {
                fv_ind.push_back(i);
            }
        }
    }

    // v[ind]
    static Vector subvec(const Vector& v, const IndexSet& ind)
    {
        const int nsub = ind.size();
        Vector res(nsub);
        for(int i = 0; i < nsub; i++)
            res[i] = v[ind[i]];
        return res;
    }

    // v[ind] = rhs
    static void subvec_assign(Vector& v, const IndexSet& ind, const Vector& rhs)
    {
        const int nsub = ind.size();
        for(int i = 0; i < nsub; i++)
            v[ind[i]] = rhs[i];
    }

public:
    // bfgs:    An object that represents the BFGS approximation matrix.
    // x0:      Current parameter vector.
    // xcp:     Computed generalized Cauchy point.
    // g:       Gradient at x0.
    // lb:      Lower bounds for x.
    // ub:      Upper bounds for x.
    // Wd:      W'(xcp - x0)
    // act_set: Active set.
    // fv_set:  Free variable set.
    // maxit:   Maximum number of iterations.
    // drt:     The output direction vector, drt = xsm - x0.
    static void subspace_minimize(
        const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& xcp, const Vector& g,
        const Vector& lb, const Vector& ub, const Vector& Wd, const IndexSet& act_set, const IndexSet& fv_set, int maxit,
        Vector& drt
    )
    {
        // std::cout << "========================= Entering subspace minimization =========================\n\n";

        // d = xcp - x0
        drt.noalias() = xcp - x0;
        // Size of active set and size of free variables
        const int nact = act_set.size();
        const int nfree = fv_set.size();
        // If there is no free variable, simply return drt
        if(nfree < 1)
        {
            // std::cout << "========================= (Early) leaving subspace minimization =========================\n\n";
            return;
        }

        // std::cout << "Active set = [ "; for(std::size_t i = 0; i < act_set.size(); i++)  std::cout << act_set[i] << " "; std::cout << "]\n";
        // std::cout << "Free variable set = [ "; for(std::size_t i = 0; i < fv_set.size(); i++)  std::cout << fv_set[i] << " "; std::cout << "]\n\n";

        // Compute b = A'd
        Vector vecb(nact);
        for(int i = 0; i < nact; i++)
            vecb[i] = drt[act_set[i]];
        // Compute F'BAb = -F'WMW'AA'd
        Vector vecc(nfree);
        bfgs.compute_FtBAb(fv_set, act_set, Wd, drt, vecb, vecc);
        // Set the vector y=F'd containing free variables, vector c=F'BAb+F'g for linear term,
        // and vectors l and u for the new bounds
        Vector vecy(nfree), vecl(nfree), vecu(nfree);
        for(int i = 0; i < nfree; i++)
        {
            const int coord = fv_set[i];
            vecy[i] = drt[coord];
            vecl[i] = lb[coord] - x0[coord];
            vecu[i] = ub[coord] - x0[coord];
            vecc[i] += g[coord];
        }

        // Dual variables
        Vector lambda = Vector::Zero(nfree), mu = Vector::Zero(nfree);

        // Iterations
        IndexSet L_set, U_set, P_set, yL_set, yU_set, yP_set;
        int k;
        for(k = 0; k < maxit; k++)
        {
            // Construct the L, U, and P sets, and then update values
            // Indices in original drt vector
            L_set.clear();
            U_set.clear();
            P_set.clear();
            // Indices in y
            yL_set.clear();
            yU_set.clear();
            yP_set.clear();
            for(int i = 0; i < nfree; i++)
            {
                const int coord = fv_set[i];
                const Scalar li = vecl[i], ui = vecu[i];
                if( (vecy[i] < li) || (vecy[i] == li && lambda[i] >= Scalar(0)) )
                {
                    L_set.push_back(coord);
                    yL_set.push_back(i);
                    vecy[i] = li;
                    mu[i] = Scalar(0);
                } else if( (vecy[i] > ui) || (vecy[i] == ui && mu[i] >= Scalar(0)) ) {
                    U_set.push_back(coord);
                    yU_set.push_back(i);
                    vecy[i] = ui;
                    lambda[i] = Scalar(0);
                } else {
                    P_set.push_back(coord);
                    yP_set.push_back(i);
                    lambda[i] = Scalar(0);
                    mu[i] = Scalar(0);
                }
            }

            /* std::cout << "** Iter " << k << " **\n";
            std::cout << "   L = [ "; for(std::size_t i = 0; i < L_set.size(); i++)  std::cout << L_set[i] << " "; std::cout << "]\n";
            std::cout << "   U = [ "; for(std::size_t i = 0; i < U_set.size(); i++)  std::cout << U_set[i] << " "; std::cout << "]\n";
            std::cout << "   P = [ "; for(std::size_t i = 0; i < P_set.size(); i++)  std::cout << P_set[i] << " "; std::cout << "]\n\n"; */

            // Split the W matrix according to P, L, and U
            Matrix WP, WL, WU;
            bfgs.split_W(P_set, L_set, U_set, WP, WL, WU);
            // Solve y[P] = -inv(B[P, P]) * (B[P, L] * l[L] + B[P, U] * u[U] + c[P])
            const int nP = P_set.size();
            if(nP > 0)
            {
                Vector rhs = subvec(vecc, yP_set);
                Vector lL = subvec(vecl, yL_set);
                Vector uU = subvec(vecu, yU_set);
                Vector tmp(nP);
                bfgs.apply_PtBQv(WP, WL, lL, tmp);
                rhs.noalias() += tmp;
                bfgs.apply_PtBQv(WP, WU, uU, tmp);
                rhs.noalias() += tmp;

                bfgs.solve_PtBP(WP, -rhs, tmp);
                subvec_assign(vecy, yP_set, tmp);
            }

            // Solve lambda[L] = B[L, F] * y + c[L]
            const int nL = L_set.size();
            const int nU = U_set.size();
            Vector Fy;
            if(nL > 0 || nU > 0)
                bfgs.apply_WtPv(fv_set, vecy, Fy);
            if(nL > 0)
            {
                Vector res;
                bfgs.apply_PtWMv(WL, Fy, res, Scalar(-1));
                res.noalias() += subvec(vecc, yL_set);
                subvec_assign(lambda, yL_set, res);
            }

            // Solve mu[U] = -B[U, F] * y - c[U]
            if(nU > 0)
            {
                Vector res;
                bfgs.apply_PtWMv(WU, Fy, res, Scalar(-1));
                res.noalias() = -res - subvec(vecc, yU_set);
                subvec_assign(mu, yU_set, res);
            }

            // Test convergence
            bool converged = true;
            for(int i = 0; i < nP; i++)
            {
                if(vecy[i] < vecl[i] || vecy[i] > vecu[i])
                {
                    converged = false;
                    break;
                }
            }
            if(!converged)
                continue;

            for(int i = 0; i < nL; i++)
            {
                if(lambda[i] < Scalar(0))
                {
                    converged = false;
                    break;
                }
            }
            if(!converged)
                continue;
            
            for(int i = 0; i < nU; i++)
            {
                if(mu[i] < Scalar(0))
                {
                    converged = false;
                    break;
                }
            }
            if(converged)
                break;
        }

        if(k >= maxit)
            throw std::runtime_error("the subspace minimization routine reached the maximum number of iterations");

        // std::cout << "** Minimization finished in " << k + 1 << " iteration(s) **\n\n";
        // std::cout << "========================= Leaving subspace minimization =========================\n\n";

        subvec_assign(drt, fv_set, vecy);
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // SUBSPACE_MIN_H
