// Copyright (C) 2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef SUBSPACE_MIN_H
#define SUBSPACE_MIN_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>
#include "LBFGS/BFGSMat.h"


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
    typedef std::vector<int> IntVector;

    // Correct out-of-bound values, and record the indices of active set and free variables
    static void analyze_boundary(Vector& x, const Vector& lb, const Vector& ub, IntVector& act_ind, IntVector& fv_ind)
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
    static Vector subvec(const Vector& v, const IntVector& ind)
    {
        const int nsub = ind.size();
        Vector res(nsub);
        for(int i = 0; i < nsub; i++)
            res[i] = v[ind[i]];
        return res;
    }

    // v[ind] = rhs
    static void subvec_assign(Vector& v, const IntVector& ind, const Vector& rhs)
    {
        const int nsub = ind.size();
        for(int i = 0; i < nsub; i++)
            v[ind[i]] = rhs[i];
    }

public:
    // bfgs:  An object that represents the BFGS approximation matrix.
    // x0:    Current parameter vector.
    // xcp:   Computed generalized Cauchy point.
    // g:     Gradient at x0.
    // lb:    Lower bounds for x.
    // ub:    Upper bounds for x.
    // maxit: Maximum number of iterations.
    // drt:   The output direction vector, drt = xsm - x0.
    static void subspace_minimize(
        const BFGSMat<Scalar, true>& bfgs, const Vector& x0, const Vector& xcp, const Vector& g,
        const Vector& lb, const Vector& ub, int maxit, Vector& drt
    )
    {
        std::cout << "========================= Entering subspace minimization =========================\n\n";

        // Get the active set and free variable set
        IntVector act_set, fv_set;
        drt.noalias() = xcp;
        analyze_boundary(drt, lb, ub, act_set, fv_set);
        // d = xcp - x0
        drt.noalias() -= x0;
        // Size of active set and size of free variables
        const int nact = act_set.size();
        const int nfree = fv_set.size();
        // If there is no free variable, simply return drt
        if(nfree < 1)
        {
            std::cout << "========================= (Early) leaving subspace minimization =========================\n\n";
            return;
        }

        std::cout << "Active set = [ "; for(std::size_t i = 0; i < act_set.size(); i++)  std::cout << act_set[i] << " "; std::cout << "]\n";
        std::cout << "Free variable set = [ "; for(std::size_t i = 0; i < fv_set.size(); i++)  std::cout << fv_set[i] << " "; std::cout << "]\n\n";
        
        // Compute b = A'd
        Vector vecb(nact);
        for(int i = 0; i < nact; i++)
            vecb[i] = drt[act_set[i]];
        // Compute F'BAb
        Vector vecc(nfree);
        bfgs.apply_PtBQv(fv_set, act_set, vecb, vecc);
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
        IntVector L_set, U_set, P_set, yL_set, yU_set, yP_set;
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

            std::cout << "** Iter " << k << " **\n";
            std::cout << "   L = [ "; for(std::size_t i = 0; i < L_set.size(); i++)  std::cout << L_set[i] << " "; std::cout << "]\n";
            std::cout << "   U = [ "; for(std::size_t i = 0; i < U_set.size(); i++)  std::cout << U_set[i] << " "; std::cout << "]\n";
            std::cout << "   P = [ "; for(std::size_t i = 0; i < P_set.size(); i++)  std::cout << P_set[i] << " "; std::cout << "]\n\n";

            // Solve y[P] = -inv(B[P, P]) * (B[P, L] * l[L] + B[P, U] * u[U] + c[P])
            const int nP = P_set.size();
            const Scalar theta = bfgs.theta();
            if(nP > 0)
            {
                Vector rhs = subvec(vecc, yP_set);
                Vector lL = subvec(vecl, yL_set);
                Vector uU = subvec(vecu, yU_set);
                Vector tmp(nP);
                bfgs.apply_PtBQv(P_set, L_set, lL, tmp);
                rhs.noalias() += tmp;
                bfgs.apply_PtBQv(P_set, U_set, uU, tmp);
                rhs.noalias() += tmp;

                if(bfgs.num_corrections() < 1)
                {
                    Vector res = -rhs / theta;
                    subvec_assign(vecy, yP_set, res);
                } else {
                    Matrix WP = bfgs.Wb(P_set);
                    Matrix middle = bfgs.Minv() - (WP.transpose() * WP) / theta;
                    Vector res = -rhs / theta - (WP * middle.lu().solve(WP.transpose() * rhs)) / (theta * theta);
                    subvec_assign(vecy, yP_set, res);
                }
            }

            // Solve lambda[L] = B[L, F] * y + c[L]
            const int nL = L_set.size();
            if(nL > 0)
            {
                Vector res;
                bfgs.apply_PtBQv(L_set, fv_set, vecy, res);
                res.noalias() += subvec(vecc, yL_set);
                subvec_assign(lambda, yL_set, res);
            }

            // Solve mu[U] = -B[U, F] * y - c[U]
            const int nU = U_set.size();
            if(nU > 0)
            {
                Vector res;
                bfgs.apply_PtBQv(U_set, fv_set, vecy, res);
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

        std::cout << "** Minimization finished in " << k + 1 << " iteration(s) **\n\n";
        std::cout << "========================= Leaving subspace minimization =========================\n\n";

        subvec_assign(drt, fv_set, vecy);
    }
};


} // namespace LBFGSpp

/// \endcond

#endif // SUBSPACE_MIN_H
