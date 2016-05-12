/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>

#include "types/simple.h"
#include "gromacs/math/utilities.h"
#include "vec.h"
#include "typedefs.h"
#include "force.h"
#include "nbnxn_kernel_gpu_ref.h"
#include "../nbnxn_consts.h"
#include "nbnxn_kernel_common.h"

#define MIC_NCL_PER_SUPERCL         8
#define MIC_CL_SIZE                 8

#pragma offload_attribute(push,target(mic))

nbnxn_pairlist_t  *mic_nbl;
nbnxn_atomdata_t  *mic_nbat;
interaction_const_t  *mic_iconst;
rvec               *mic_shift_vec;
real  *                     mic_f;
real  *				mic_t_f;
real  *				mic_t_fshift;
real  *				cpu_t_f;
real  *				cpu_t_fshift;
real  *                     mic_fshift;
real  *                     mic_Vc;
real  *                     mic_Vvdw;
// nbnxn_sci_t  				*mic_nbln;
real         				*mic_nbat_x;
real         				*mic_ic_tabq_coul_F;
real       *        mic_nbat_nbfp;
int       *         mic_shift;
int       *         mic_nbat_type;
nbnxn_sci_t *		mic_nbl_sci;
nbnxn_cj4_t *		mic_nbl_cj4;
nbnxn_excl_t *		mic_nbl_excl;
int 	SCI_ALLOC = 0;
int 	CJ4_ALLOC = 0;
int 	EXCL_ALLOC = 0;
int     mic_nthreads = 112;
unsigned int SCI_LOC = 0x0;
unsigned int CJ4_LOC = 0x0;
unsigned int EXCL_LOC = 0x0;
real	*sig_mic_x;
real	*sig_mic_type;
real	*sig_mic_sci;
real  *                     mic_fshift1;
real  *                     mic_Vc1;
real  *                     mic_Vvdw1;
real  *                     mic_f1;
real  *                     mic_fshift2;
real  *                     mic_Vc2;
real  *                     mic_Vvdw2;
real  *                     mic_f2;
real  *                     cpu_fshift;
real  *                     cpu_Vc;
real  *                     cpu_Vvdw;
real  *                     cpu_f;
int							cpu_nthreads;
real  *                     local_mic_f;
real  *                     non_local_mic_f;
real  *				local_mic_t_f;
real  *				local_mic_t_fshift;
real  *				local_cpu_t_f;
real  *				local_cpu_t_fshift;
real  *				non_local_mic_t_f;
real  *				non_local_mic_t_fshift;
real  *				non_local_cpu_t_f;
real  *				non_local_cpu_t_fshift;
real         				*local_mic_nbat_x;
real         				*non_local_mic_nbat_x;
int       *         local_mic_nbat_type;
int       *         non_local_mic_nbat_type;
int 	local_mic_nalloc = 0;
int 	non_local_mic_nalloc = 0;
static const float
    tinyf =  1e-30,
    halff =  5.0000000000e-01, /* 0x3F000000 */
    onef  =  1.0000000000e+00, /* 0x3F800000 */
    twof  =  2.0000000000e+00, /* 0x40000000 */
/* c = (subfloat)0.84506291151 */
    erxf =  8.4506291151e-01,  /* 0x3f58560b */
/*
 * Coefficients for approximation to  erf on [0,0.84375]
 */
    efxf  =  1.2837916613e-01, /* 0x3e0375d4 */
    efx8f =  1.0270333290e+00, /* 0x3f8375d4 */
    pp0f  =  1.2837916613e-01, /* 0x3e0375d4 */
    pp1f  = -3.2504209876e-01, /* 0xbea66beb */
    pp2f  = -2.8481749818e-02, /* 0xbce9528f */
    pp3f  = -5.7702702470e-03, /* 0xbbbd1489 */
    pp4f  = -2.3763017452e-05, /* 0xb7c756b1 */
    qq1f  =  3.9791721106e-01, /* 0x3ecbbbce */
    qq2f  =  6.5022252500e-02, /* 0x3d852a63 */
    qq3f  =  5.0813062117e-03, /* 0x3ba68116 */
    qq4f  =  1.3249473704e-04, /* 0x390aee49 */
    qq5f  = -3.9602282413e-06, /* 0xb684e21a */
/*
 * Coefficients for approximation to  erf  in [0.84375,1.25]
 */
    pa0f = -2.3621185683e-03,  /* 0xbb1acdc6 */
    pa1f =  4.1485610604e-01,  /* 0x3ed46805 */
    pa2f = -3.7220788002e-01,  /* 0xbebe9208 */
    pa3f =  3.1834661961e-01,  /* 0x3ea2fe54 */
    pa4f = -1.1089469492e-01,  /* 0xbde31cc2 */
    pa5f =  3.5478305072e-02,  /* 0x3d1151b3 */
    pa6f = -2.1663755178e-03,  /* 0xbb0df9c0 */
    qa1f =  1.0642088205e-01,  /* 0x3dd9f331 */
    qa2f =  5.4039794207e-01,  /* 0x3f0a5785 */
    qa3f =  7.1828655899e-02,  /* 0x3d931ae7 */
    qa4f =  1.2617121637e-01,  /* 0x3e013307 */
    qa5f =  1.3637083583e-02,  /* 0x3c5f6e13 */
    qa6f =  1.1984500103e-02,  /* 0x3c445aa3 */
/*
 * Coefficients for approximation to  erfc in [1.25,1/0.35]
 */
    ra0f = -9.8649440333e-03,  /* 0xbc21a093 */
    ra1f = -6.9385856390e-01,  /* 0xbf31a0b7 */
    ra2f = -1.0558626175e+01,  /* 0xc128f022 */
    ra3f = -6.2375331879e+01,  /* 0xc2798057 */
    ra4f = -1.6239666748e+02,  /* 0xc322658c */
    ra5f = -1.8460508728e+02,  /* 0xc3389ae7 */
    ra6f = -8.1287437439e+01,  /* 0xc2a2932b */
    ra7f = -9.8143291473e+00,  /* 0xc11d077e */
    sa1f =  1.9651271820e+01,  /* 0x419d35ce */
    sa2f =  1.3765776062e+02,  /* 0x4309a863 */
    sa3f =  4.3456588745e+02,  /* 0x43d9486f */
    sa4f =  6.4538726807e+02,  /* 0x442158c9 */
    sa5f =  4.2900814819e+02,  /* 0x43d6810b */
    sa6f =  1.0863500214e+02,  /* 0x42d9451f */
    sa7f =  6.5702495575e+00,  /* 0x40d23f7c */
    sa8f = -6.0424413532e-02,  /* 0xbd777f97 */
/*
 * Coefficients for approximation to  erfc in [1/.35,28]
 */
    rb0f = -9.8649431020e-03,  /* 0xbc21a092 */
    rb1f = -7.9928326607e-01,  /* 0xbf4c9dd4 */
    rb2f = -1.7757955551e+01,  /* 0xc18e104b */
    rb3f = -1.6063638306e+02,  /* 0xc320a2ea */
    rb4f = -6.3756646729e+02,  /* 0xc41f6441 */
    rb5f = -1.0250950928e+03,  /* 0xc480230b */
    rb6f = -4.8351919556e+02,  /* 0xc3f1c275 */
    sb1f =  3.0338060379e+01,  /* 0x41f2b459 */
    sb2f =  3.2579251099e+02,  /* 0x43a2e571 */
    sb3f =  1.5367296143e+03,  /* 0x44c01759 */
    sb4f =  3.1998581543e+03,  /* 0x4547fdbb */
    sb5f =  2.5530502930e+03,  /* 0x451f90ce */
    sb6f =  4.7452853394e+02,  /* 0x43ed43a7 */
    sb7f = -2.2440952301e+01;  /* 0xc1b38712 */
	
float mic_erff(float x)
{
    __int32 hx, ix, i;
    float       R, S, P, Q, s, y, z, r;

    union
    {
        float  f;
        int    i;
    }
    conv;

    conv.f = x;
    hx     = conv.i;

    ix = hx&0x7fffffff;
    if (ix >= 0x7f800000)
    {
        /* erf(nan)=nan */
        i = ((__int32)hx>>31)<<1;
        return (float)(1-i)+onef/x; /* erf(+-inf)=+-1 */
    }

    if (ix < 0x3f580000)
    {
        /* |x|<0.84375 */
        if (ix < 0x31800000)
        {
            /* |x|<2**-28 */
            if (ix < 0x04000000)
            {
                return (float)0.125*((float)8.0*x+efx8f*x);             /*avoid underflow */
            }
            return x + efxf*x;
        }
        z = x*x;
        r = pp0f+z*(pp1f+z*(pp2f+z*(pp3f+z*pp4f)));
        s = onef+z*(qq1f+z*(qq2f+z*(qq3f+z*(qq4f+z*qq5f))));
        y = r/s;
        return x + x*y;
    }
    if (ix < 0x3fa00000)
    {
        /* 0.84375 <= |x| < 1.25 */
        s = fabs(x)-onef;
        P = pa0f+s*(pa1f+s*(pa2f+s*(pa3f+s*(pa4f+s*(pa5f+s*pa6f)))));
        Q = onef+s*(qa1f+s*(qa2f+s*(qa3f+s*(qa4f+s*(qa5f+s*qa6f)))));
        if (hx >= 0)
        {
            return erxf + P/Q;
        }
        else
        {
            return -erxf - P/Q;
        }
    }
    if (ix >= 0x40c00000)
    {
        /* inf>|x|>=6 */
        if (hx >= 0)
        {
            return onef-tinyf;
        }
        else
        {
            return tinyf-onef;
        }
    }
    x = fabs(x);
    s = onef/(x*x);
    if (ix < 0x4036DB6E)
    {
        /* |x| < 1/0.35 */
        R = ra0f+s*(ra1f+s*(ra2f+s*(ra3f+s*(ra4f+s*(ra5f+s*(ra6f+s*ra7f))))));
        S = onef+s*(sa1f+s*(sa2f+s*(sa3f+s*(sa4f+s*(sa5f+s*(sa6f+s*(sa7f+s*sa8f)))))));
    }
    else
    {
        /* |x| >= 1/0.35 */
        R = rb0f+s*(rb1f+s*(rb2f+s*(rb3f+s*(rb4f+s*(rb5f+s*rb6f)))));
        S = onef+s*(sb1f+s*(sb2f+s*(sb3f+s*(sb4f+s*(sb5f+s*(sb6f+s*sb7f))))));
    }

    conv.f = x;
    conv.i = conv.i & 0xfffff000;
    z      = conv.f;

    r  =  exp(-z*z-(float)0.5625)*exp((z-x)*(z+x)+R/S);
    if (hx >= 0)
    {
        return onef-r/x;
    }
    else
    {
        return r/x-onef;
    }
}

void mic_kernel(const nbnxn_pairlist_t     *nbl,
                     const nbnxn_atomdata_t     *nbat,
                     const interaction_const_t  *iconst,
                     rvec                       *shift_vec,
                     int                         force_flags,
                     int                         clearF,
                     real  *                     f,
                     real  *                     fshift,
                     real  *                     Vc,
                     real  *                     Vvdw,
					 int						 device_id)
{
	gmx_bool            mic_bEner;
	gmx_bool            mic_bEwald;
	real                rcut2, rvdw2, rlist2;
	int                 ntype;
	int       *         type;
	real                facel;
	real       *  shiftvec;
	real       *        vdwparam;
	real         *x;
	real         *Ftab = NULL;
	int 				n;	
		
	int                 npair_tot;
	int                 nhwu, nhwu_pruned;
	real 				t_Vc = 0, t_Vvdw = 0, f_sum = 0.0;
	
	
	mic_bEner = (force_flags & GMX_FORCE_ENERGY);

	mic_bEwald = EEL_FULL(iconst->eeltype);
	// if (clearF == enbvClearFYes)
	// {
		memset(f,0,nbat->natoms*nbat->fstride*sizeof(real));
	// }
	memset(fshift,0,SHIFTS*DIM*sizeof(real));
	if(mic_bEwald)
	{
		Ftab = iconst->tabq_coul_F;
	}
	rcut2               = iconst->rcoulomb*iconst->rcoulomb;
	rvdw2               = iconst->rvdw*iconst->rvdw;

	rlist2              = nbl->rlist*nbl->rlist;

	type                = nbat->type;
	facel               = iconst->epsfac;
	shiftvec            = shift_vec[0];
	vdwparam            = nbat->nbfp;
	ntype               = nbat->ntype;

	x = nbat->x;
	npair_tot   = 0;
	nhwu        = 0;
	nhwu_pruned = 0;

	#pragma omp parallel for schedule(static)num_threads(mic_nthreads)
	for (n = 0; n < mic_nthreads; n++)
	{
		real				*t_f;
		real				*t_fshift;

		t_f	=	mic_t_f + nbat->nalloc*nbat->fstride*n;
		t_fshift	=	mic_t_fshift + SHIFTS*DIM*n;
		memset(t_f,0,nbat->natoms*nbat->fstride*sizeof(real));
		memset(t_fshift,0,SHIFTS*DIM*sizeof(real));
	}
	// printf("mic nbat->natoms*nbat->fstride:%d x[nbat->nbat->natoms*nbat->fstride]:%f\n",nbat->natoms*nbat->fstride,x[nbat->natoms*nbat->fstride]);
	// fflush(0);
	// int max_ifs=0,max_jfs=0;
	#pragma omp parallel for schedule(static)num_threads(mic_nthreads)reduction(+:t_Vc,t_Vvdw)
	for (n = nbl->nsci*(device_id)/4; n < nbl->nsci*(device_id+1)/4; n++)
	{
		int                 i;
		int                 ish3;
		int                 sci;
		int                 cj4_ind0, cj4_ind1, cj4_ind;
		int                 ci, cj;
		int                 ic, jc, ia, ja, is, ifs, js, jfs, im, jm;
		int                 n0;
		int                 ggid;
		real                shX, shY, shZ;
		real                fscal, tx, ty, tz;
		real                rinvsq;
		real                iq;
		real                qq, vcoul = 0, krsq, vctot;
		int                 nti;
		int                 tj;
		real                rt, r, eps;
		real                rinvsix;
		real                Vvdwtot;
		real                Vvdw_rep, Vvdw_disp;
		real                ix, iy, iz, fix, fiy, fiz;
		real                jx, jy, jz;
		real                dx, dy, dz, rsq, rinv;
		int                 int_bit, npair;
		real                fexcl;
		real                c6, c12;
		nbnxn_sci_t  *nbln;
		const nbnxn_excl_t *excl[2];
		real				*t_f;
		real				*t_fshift;
		
		
		t_f	=	mic_t_f + nbat->nalloc*nbat->fstride*omp_get_thread_num();
		t_fshift	=	mic_t_fshift + SHIFTS*DIM*omp_get_thread_num();
		nbln = &nbl->sci[n];

		ish3             = 3*nbln->shift;
		shX              = shiftvec[ish3];
		shY              = shiftvec[ish3+1];
		shZ              = shiftvec[ish3+2];
		cj4_ind0         = nbln->cj4_ind_start;
		cj4_ind1         = nbln->cj4_ind_end;
		sci              = nbln->sci;
		vctot            = 0;
		Vvdwtot          = 0;
		if (nbln->shift == CENTRAL &&
			nbl->cj4[cj4_ind0].cj[0] == sci*MIC_NCL_PER_SUPERCL)
		{
			/* we have the diagonal:
			 * add the charge self interaction energy term
			 */
			for (im = 0; im < MIC_NCL_PER_SUPERCL; im++)
			{
				ci = sci*MIC_NCL_PER_SUPERCL + im;
				for (ic = 0; ic < MIC_CL_SIZE; ic++)
				{
					ia     = ci*MIC_CL_SIZE + ic;
					iq     = x[ia*nbat->xstride+3];
					vctot += iq*iq;
				}
			}
			
			if (!mic_bEwald)
			{
				vctot *= -facel*0.5*iconst->c_rf; 
			}
			else
			{
				/* last factor 1/sqrt(pi) */
				vctot *= -facel*iconst->ewaldcoeff_q*M_1_SQRTPI;
			}
		}
		
		for (cj4_ind = cj4_ind0; (cj4_ind < cj4_ind1); cj4_ind++)
		{
			excl[0]           = &nbl->excl[nbl->cj4[cj4_ind].imei[0].excl_ind];
			excl[1]           = &nbl->excl[nbl->cj4[cj4_ind].imei[1].excl_ind];

			for (jm = 0; jm < NBNXN_GPU_JGROUP_SIZE; jm++)
			{
				cj               = nbl->cj4[cj4_ind].cj[jm];
				

				for (im = 0; im < MIC_NCL_PER_SUPERCL; im++)
				{
					/* We're only using the first imask,
					 * but here imei[1].imask is identical.
					 */
					if ((nbl->cj4[cj4_ind].imei[0].imask >> (jm*MIC_NCL_PER_SUPERCL+im)) & 1)
					{

						ci               = sci*MIC_NCL_PER_SUPERCL + im;


						for (ic = 0; ic < MIC_CL_SIZE; ic++)
						{
							ia               = ci*MIC_CL_SIZE + ic;

							is               = ia*nbat->xstride;
							ifs              = ia*nbat->fstride;
							// if(ifs>max_ifs)max_ifs = ifs;
							ix               = shX + x[is+0];
							iy               = shY + x[is+1];
							iz               = shZ + x[is+2];
							iq               = facel*x[is+3];
							nti              = ntype*2*type[ia];
							fix              = 0;
							fiy              = 0;
							fiz              = 0;

							for (jc = 0; jc < MIC_CL_SIZE; jc++)
							{
								ja               = cj*MIC_CL_SIZE + jc;
								if (nbln->shift == CENTRAL &&
									ci == cj && ja <= ia)
								{
									continue;
								}

								int_bit = ((excl[jc>>2]->pair[(jc & 3)*MIC_CL_SIZE+ic] >> (jm*MIC_NCL_PER_SUPERCL+im)) & 1);

								js               = ja*nbat->xstride;
								jfs              = ja*nbat->fstride;
								// if(jfs>max_jfs)max_jfs = jfs;
								jx               = x[js+0];
								jy               = x[js+1];
								jz               = x[js+2];
								dx               = ix - jx;
								dy               = iy - jy;
								dz               = iz - jz;
								rsq              = dx*dx + dy*dy + dz*dz;
								if (rsq >= rcut2)
								{
									continue;
								}
								rsq             += (1.0 - int_bit)*NBNXN_AVOID_SING_R2_INC;

								rinv             =  (1.0/sqrtf(rsq));
								rinvsq           = rinv*rinv;
								fscal            = 0;

								qq               = iq*x[js+3];
								if (!mic_bEwald)
								{
									/* Reaction-field */
									krsq  = iconst->k_rf*rsq;
									fscal = qq*(int_bit*rinv - 2*krsq)*rinvsq;
									if (mic_bEner)
									{
										vcoul = qq*(int_bit*rinv + krsq - iconst->c_rf);
									}
								}
								else
								{
									r     = rsq*rinv;
									rt    = r*iconst->tabq_scale;
									n0    = rt;
									eps   = rt - n0;

									fexcl = (1 - eps)*Ftab[n0] + eps*Ftab[n0+1];

									fscal = qq*(int_bit*rinvsq - fexcl)*rinv;

									if (mic_bEner)
									{
										vcoul = qq*((int_bit - mic_erff(iconst->ewaldcoeff_q*r))*rinv - int_bit*iconst->sh_ewald);
									}
								}

								if (rsq < rvdw2)
								{
									tj        = nti + 2*type[ja];

									/* Vanilla Lennard-Jones cutoff */
									c6        = vdwparam[tj];
									c12       = vdwparam[tj+1];

									rinvsix   = int_bit*rinvsq*rinvsq*rinvsq;
									Vvdw_disp = c6*rinvsix;
									Vvdw_rep  = c12*rinvsix*rinvsix;
									fscal    += (Vvdw_rep - Vvdw_disp)*rinvsq;

									if (mic_bEner)
									{
										vctot   += vcoul;
										
										Vvdwtot +=
											(Vvdw_rep - int_bit*c12*iconst->sh_invrc6*iconst->sh_invrc6)/12 -
											(Vvdw_disp - int_bit*c6*iconst->sh_invrc6)/6;
									}
								}
								tx        = fscal*dx;
								ty        = fscal*dy;
								tz        = fscal*dz;
								fix       = fix + tx;
								fiy       = fiy + ty;
								fiz       = fiz + tz;
								t_f[jfs+0] -= tx;
								t_f[jfs+1] -= ty;
								// if(jfs==117336)printf("mic fiz:%f\n",fiz);
								// fflush(0);
								t_f[jfs+2] -= tz;
								
							}
							t_f[ifs+0]        += fix;
							t_f[ifs+1]        += fiy;
							// if(ifs==117336)printf("mic fiz:%f\n",fiz);
							// fflush(0);
							t_f[ifs+2]        += fiz;
							t_fshift[ish3]     = t_fshift[ish3]   + fix;
							t_fshift[ish3+1]   = t_fshift[ish3+1] + fiy;
							t_fshift[ish3+2]   = t_fshift[ish3+2] + fiz;
						}
					}
				}
			}
		}
		if (mic_bEner)
		{
			t_Vc         = t_Vc   + vctot;
			t_Vvdw     = t_Vvdw + Vvdwtot;
		}
	}			
	// printf("mic max_ifs:%d max_jfs:%d\n",max_ifs,max_jfs);
	Vc[0] = t_Vc;
	Vvdw[0] = t_Vvdw;
	int l,j,part;
	part = nbat->natoms*nbat->fstride/(mic_nthreads-1);
	#pragma omp parallel for schedule(static)num_threads(mic_nthreads)
	for(j = 0;j < mic_nthreads;j++){
		int i,k;
		if(j == mic_nthreads-1){
			for(k = 0;k < mic_nthreads;k ++){
				for(i = part*(mic_nthreads-1);i < nbat->natoms*nbat->fstride;i ++){
					f[i] += mic_t_f[k * nbat->nalloc*nbat->fstride + i];
				}
			}
		}
		else{
			for(k = 0;k < mic_nthreads;k++){
				for(i = 0;i < part;i ++){
					f[part*j+i] += mic_t_f[k * nbat->nalloc*nbat->fstride + part*j+ i];
				}
			}
		}
	
		
	}
	for(j = 0;j < mic_nthreads;j++){
		for(l = 0;l < SHIFTS*DIM;l ++){
			fshift[l] += mic_t_fshift[j * SHIFTS*DIM + l];
		}
	}
}

#pragma offload_attribute(pop)

void cpu_kernel(const nbnxn_pairlist_t     *nbl,
                     const nbnxn_atomdata_t     *nbat,
                     const interaction_const_t  *iconst,
                     rvec                       *shift_vec,
                     int                         force_flags,
                     int                         clearF,
                     real  *                     f,
                     real  *                     fshift,
                     real  *                     Vc,
                     real  *                     Vvdw)
{
	gmx_bool            cpu_bEner;
	gmx_bool            cpu_bEwald;
	real                rcut2, rvdw2, rlist2;
	int                 ntype;
	int       *         type;
	real                facel;
	real       *  shiftvec;
	real       *        vdwparam;
	real         *x;
	real         *Ftab = NULL;
	int 				n;	
		
	int                 npair_tot;
	int                 nhwu, nhwu_pruned;
	real 				t_Vc = 0, t_Vvdw = 0, f_sum = 0.0;
	
	
	cpu_bEner = (force_flags & GMX_FORCE_ENERGY);

	cpu_bEwald = EEL_FULL(iconst->eeltype);
	// if (clearF == enbvClearFYes)
	// {
		memset(f,0,nbat->natoms*nbat->fstride*sizeof(real));
	// }
	memset(fshift,0,SHIFTS*DIM*sizeof(real));
	if(cpu_bEwald)
	{
		Ftab = iconst->tabq_coul_F;
	}
	rcut2               = iconst->rcoulomb*iconst->rcoulomb;
	rvdw2               = iconst->rvdw*iconst->rvdw;

	rlist2              = nbl->rlist*nbl->rlist;

	type                = nbat->type;
	facel               = iconst->epsfac;
	shiftvec            = shift_vec[0];
	vdwparam            = nbat->nbfp;
	ntype               = nbat->ntype;

	x = nbat->x;
	npair_tot   = 0;
	nhwu        = 0;
	nhwu_pruned = 0;

	#pragma omp parallel for schedule(static)num_threads(cpu_nthreads)
	for (n = 0; n < cpu_nthreads; n++)
	{
		real				*t_f;
		real				*t_fshift;

		t_f	=	cpu_t_f + nbat->nalloc*nbat->fstride*n;
		t_fshift	=	cpu_t_fshift + SHIFTS*DIM*n;
		memset(t_f,0,nbat->natoms*nbat->fstride*sizeof(real));
		memset(t_fshift,0,SHIFTS*DIM*sizeof(real));
	}
	// printf("cpu nbat->natoms*nbat->fstride:%d x[nbat->natoms*nbat->fstride]:%f\n",nbat->natoms*nbat->fstride,x[nbat->natoms*nbat->fstride]);
	// fflush(0);
	// int max_ifs=0,max_jfs=0;
	#pragma omp parallel for schedule(static)num_threads(cpu_nthreads)reduction(+:t_Vc,t_Vvdw)
	for (n = nbl->nsci*3/4 ; n < nbl->nsci; n++)
	{
		int                 i;
		int                 ish3;
		int                 sci;
		int                 cj4_ind0, cj4_ind1, cj4_ind;
		int                 ci, cj;
		int                 ic, jc, ia, ja, is, ifs, js, jfs, im, jm;
		int                 n0;
		int                 ggid;
		real                shX, shY, shZ;
		real                fscal, tx, ty, tz;
		real                rinvsq;
		real                iq;
		real                qq, vcoul = 0, krsq, vctot;
		int                 nti;
		int                 tj;
		real                rt, r, eps;
		real                rinvsix;
		real                Vvdwtot;
		real                Vvdw_rep, Vvdw_disp;
		real                ix, iy, iz, fix, fiy, fiz;
		real                jx, jy, jz;
		real                dx, dy, dz, rsq, rinv;
		int                 int_bit, npair;
		real                fexcl;
		real                c6, c12;
		nbnxn_sci_t  *nbln;
		const nbnxn_excl_t *excl[2];
		real				*t_f;
		real				*t_fshift;
		
		
		t_f	=	cpu_t_f + nbat->nalloc*nbat->fstride*omp_get_thread_num();
		t_fshift	=	cpu_t_fshift + SHIFTS*DIM*omp_get_thread_num();
		nbln = &nbl->sci[n];

		ish3             = 3*nbln->shift;
		shX              = shiftvec[ish3];
		shY              = shiftvec[ish3+1];
		shZ              = shiftvec[ish3+2];
		cj4_ind0         = nbln->cj4_ind_start;
		cj4_ind1         = nbln->cj4_ind_end;
		sci              = nbln->sci;
		vctot            = 0;
		Vvdwtot          = 0;
		if (nbln->shift == CENTRAL &&
			nbl->cj4[cj4_ind0].cj[0] == sci*MIC_NCL_PER_SUPERCL)
		{
			/* we have the diagonal:
			 * add the charge self interaction energy term
			 */
			for (im = 0; im < MIC_NCL_PER_SUPERCL; im++)
			{
				ci = sci*MIC_NCL_PER_SUPERCL + im;
				for (ic = 0; ic < MIC_CL_SIZE; ic++)
				{
					ia     = ci*MIC_CL_SIZE + ic;
					iq     = x[ia*nbat->xstride+3];
					vctot += iq*iq;
				}
			}
			
			if (!cpu_bEwald)
			{
				vctot *= -facel*0.5*iconst->c_rf; 
			}
			else
			{
				/* last factor 1/sqrt(pi) */
				vctot *= -facel*iconst->ewaldcoeff_q*M_1_SQRTPI;
			}
		}
		
		for (cj4_ind = cj4_ind0; (cj4_ind < cj4_ind1); cj4_ind++)
		{
			excl[0]           = &nbl->excl[nbl->cj4[cj4_ind].imei[0].excl_ind];
			excl[1]           = &nbl->excl[nbl->cj4[cj4_ind].imei[1].excl_ind];

			for (jm = 0; jm < NBNXN_GPU_JGROUP_SIZE; jm++)
			{
				cj               = nbl->cj4[cj4_ind].cj[jm];

				for (im = 0; im < MIC_NCL_PER_SUPERCL; im++)
				{
					/* We're only using the first imask,
					 * but here imei[1].imask is identical.
					 */
					if ((nbl->cj4[cj4_ind].imei[0].imask >> (jm*MIC_NCL_PER_SUPERCL+im)) & 1)
					{

						ci               = sci*MIC_NCL_PER_SUPERCL + im;


						for (ic = 0; ic < MIC_CL_SIZE; ic++)
						{
							ia               = ci*MIC_CL_SIZE + ic;

							is               = ia*nbat->xstride;
							ifs              = ia*nbat->fstride;
							// if(ifs>max_ifs)max_ifs = ifs;
							ix               = shX + x[is+0];
							iy               = shY + x[is+1];
							iz               = shZ + x[is+2];
							iq               = facel*x[is+3];
							nti              = ntype*2*type[ia];

							fix              = 0;
							fiy              = 0;
							fiz              = 0;
							for (jc = 0; jc < MIC_CL_SIZE; jc++)
							{
								ja               = cj*MIC_CL_SIZE + jc;
								if (nbln->shift == CENTRAL &&
									ci == cj && ja <= ia)
								{
									continue;
								}

								int_bit = ((excl[jc>>2]->pair[(jc & 3)*MIC_CL_SIZE+ic] >> (jm*MIC_NCL_PER_SUPERCL+im)) & 1);

								js               = ja*nbat->xstride;
								jfs              = ja*nbat->fstride;
								// if(jfs>max_jfs)max_jfs = jfs;
								jx               = x[js+0];
								jy               = x[js+1];
								jz               = x[js+2];
								dx               = ix - jx;
								dy               = iy - jy;
								dz               = iz - jz;
								rsq              = dx*dx + dy*dy + dz*dz;
								if (rsq >= rcut2)
								{
									continue;
								}
								rsq             += (1.0 - int_bit)*NBNXN_AVOID_SING_R2_INC;

								rinv             =  (1.0/sqrtf(rsq));
								rinvsq           = rinv*rinv;
								fscal            = 0;

								qq               = iq*x[js+3];
								if (!cpu_bEwald)
								{
									/* Reaction-field */
									krsq  = iconst->k_rf*rsq;
									fscal = qq*(int_bit*rinv - 2*krsq)*rinvsq;
									if (cpu_bEner)
									{
										vcoul = qq*(int_bit*rinv + krsq - iconst->c_rf);
									}
								}
								else
								{
									r     = rsq*rinv;
									rt    = r*iconst->tabq_scale;
									n0    = rt;
									eps   = rt - n0;

									fexcl = (1 - eps)*Ftab[n0] + eps*Ftab[n0+1];

									fscal = qq*(int_bit*rinvsq - fexcl)*rinv;

									if (cpu_bEner)
									{
										vcoul = qq*((int_bit - mic_erff(iconst->ewaldcoeff_q*r))*rinv - int_bit*iconst->sh_ewald);
									}
								}

								if (rsq < rvdw2)
								{
									tj        = nti + 2*type[ja];

									/* Vanilla Lennard-Jones cutoff */
									c6        = vdwparam[tj];
									c12       = vdwparam[tj+1];

									rinvsix   = int_bit*rinvsq*rinvsq*rinvsq;
									Vvdw_disp = c6*rinvsix;
									Vvdw_rep  = c12*rinvsix*rinvsix;
									fscal    += (Vvdw_rep - Vvdw_disp)*rinvsq;

									if (cpu_bEner)
									{
										vctot   += vcoul;
										
										Vvdwtot +=
											(Vvdw_rep - int_bit*c12*iconst->sh_invrc6*iconst->sh_invrc6)/12 -
											(Vvdw_disp - int_bit*c6*iconst->sh_invrc6)/6;
									}
								}

								tx        = fscal*dx;
								ty        = fscal*dy;
								tz        = fscal*dz;
								fix       = fix + tx;
								fiy       = fiy + ty;
								fiz       = fiz + tz;
								t_f[jfs+0] -= tx;
								t_f[jfs+1] -= ty;
								// if(jfs==117336)printf("cpu fiz:%f\n",fiz);
								// fflush(0);
								t_f[jfs+2] -= tz;
								
							}
							t_f[ifs+0]        += fix;
							t_f[ifs+1]        += fiy;
							// if(ifs==117336)printf("cpu fiz:%f\n",fiz);
							// fflush(0);
							t_f[ifs+2]        += fiz;
							t_fshift[ish3]     = t_fshift[ish3]   + fix;
							t_fshift[ish3+1]   = t_fshift[ish3+1] + fiy;
							t_fshift[ish3+2]   = t_fshift[ish3+2] + fiz;
						}
					}
				}
			}
		}
		if (cpu_bEner)
		{
			t_Vc         = t_Vc   + vctot;
			t_Vvdw     = t_Vvdw + Vvdwtot;
		}
	}			
	// printf("cpu max_ifs:%d max_jfs:%d\n",max_ifs,max_jfs);
	Vc[0] = t_Vc;
	Vvdw[0] = t_Vvdw;
	int l,j,part;
	part = nbat->natoms*nbat->fstride/(cpu_nthreads-1);
	#pragma omp parallel for schedule(static)num_threads(cpu_nthreads)
	for(j = 0;j < cpu_nthreads;j++){
		int i,k;
		if(j == cpu_nthreads-1){
			for(k = 0;k < cpu_nthreads;k ++){
				for(i = part*(cpu_nthreads-1);i < nbat->natoms*nbat->fstride;i ++){
					f[i] += cpu_t_f[k * nbat->nalloc*nbat->fstride + i];
				}
			}
		}
		else{
			for(k = 0;k < cpu_nthreads;k++){
				for(i = 0;i < part;i ++){
					f[part*j+i] += cpu_t_f[k * nbat->nalloc*nbat->fstride + part*j+ i];
				}
			}
		}
	
		
	}
	for(j = 0;j < cpu_nthreads;j++){
		for(l = 0;l < SHIFTS*DIM;l ++){
			fshift[l] += cpu_t_fshift[j * SHIFTS*DIM + l];
		}
	}
}

void
nbnxn_kernel_gpu_ref(const nbnxn_pairlist_t     *nbl,
                     const nbnxn_atomdata_t     *nbat,
                     const interaction_const_t  *iconst,
                     rvec                       *shift_vec,
                     int                         force_flags,
                     int                         clearF,
                     real  *                     f,
                     real  *                     fshift,
                     real  *                     Vc,
                     real  *                     Vvdw)
{
	gmx_bool            bEner;
	gmx_bool            bEwald;

    if (nbl->na_ci != MIC_CL_SIZE)
    {
        gmx_fatal(FARGS, "The neighborlist cluster size in the GPU reference kernel is %d, expected it to be %d", nbl->na_ci, MIC_CL_SIZE);
    }

    
	if (clearF == enbvClearFYes){
		clear_f(nbat, 0, f);
	}
    bEner = (force_flags & GMX_FORCE_ENERGY);

    bEwald = EEL_FULL(iconst->eeltype);
    if (bEwald)
    {
		mic_ic_tabq_coul_F = iconst->tabq_coul_F;
    }


	real *sig_mic1 = mic_f1, *sig_mic2 = mic_f2, *sig_mic0 = mic_f;
	int device_id=0;
	#pragma omp parallel for
	for(device_id = 0;device_id<3;device_id++){
		if(device_id == 0){
			printf("omp sig_mic0:%x \n",sig_mic0);
			#pragma offload target(mic:0)nocopy(mic_nbl),nocopy(mic_nbat),nocopy(mic_iconst),nocopy(mic_shift_vec),nocopy(mic_f),nocopy(mic_fshift),nocopy(mic_Vc),nocopy(mic_Vvdw)signal(sig_mic0)
			{
				mic_kernel(mic_nbl,mic_nbat,mic_iconst,mic_shift_vec,force_flags,clearF,mic_f,mic_fshift,mic_Vc,mic_Vvdw,device_id);
				
			}
		}
		else if(device_id == 1){
			printf("omp sig_mic1:%x \n",sig_mic1);
			#pragma offload target(mic:1)nocopy(mic_nbl),nocopy(mic_nbat),nocopy(mic_iconst),nocopy(mic_shift_vec),nocopy(mic_f1),nocopy(mic_fshift1),nocopy(mic_Vc1),nocopy(mic_Vvdw1)signal(sig_mic1)
			{
				mic_kernel(mic_nbl,mic_nbat,mic_iconst,mic_shift_vec,force_flags,clearF,mic_f1,mic_fshift1,mic_Vc1,mic_Vvdw1,device_id);
				
			}
		}
		else{
			printf("omp sig_mic2:%x\n",sig_mic2);
			 #pragma offload target(mic:2)nocopy(mic_nbl),nocopy(mic_nbat),nocopy(mic_iconst),nocopy(mic_shift_vec),nocopy(mic_f2),nocopy(mic_fshift2),nocopy(mic_Vc2),nocopy(mic_Vvdw2)signal(sig_mic2)
			{
				mic_kernel(mic_nbl,mic_nbat,mic_iconst,mic_shift_vec,force_flags,clearF,mic_f2,mic_fshift2,mic_Vc2,mic_Vvdw2,device_id);
				
			}
		}
		
	}
	
	{
		cpu_kernel(mic_nbl,mic_nbat,mic_iconst,mic_shift_vec,force_flags,clearF,cpu_f,cpu_fshift,cpu_Vc,cpu_Vvdw);
	}
	{
		#pragma offload target(mic:0) \
							out(mic_fshift:length(SHIFTS*DIM)alloc_if(0) free_if(0))\
							out(mic_Vc:length(1)alloc_if(0) free_if(0))\
							out(mic_Vvdw:length(1)alloc_if(0) free_if(0))\
							out(mic_f:length(mic_nbat->natoms*mic_nbat->fstride)alloc_if(0) free_if(0))wait(sig_mic0)\
							
							
							{
							}
		#pragma offload target(mic:1) \
							out(mic_fshift1:length(SHIFTS*DIM)alloc_if(0) free_if(0))\
							out(mic_Vc1:length(1)alloc_if(0) free_if(0))\
							out(mic_Vvdw1:length(1)alloc_if(0) free_if(0))\
							out(mic_f1:length(mic_nbat->natoms*mic_nbat->fstride)alloc_if(0) free_if(0))wait(sig_mic1)\
							
							{
							}
		#pragma offload target(mic:2) \
							out(mic_fshift2:length(SHIFTS*DIM)alloc_if(0) free_if(0))\
							out(mic_Vc2:length(1)alloc_if(0) free_if(0))\
							out(mic_Vvdw2:length(1)alloc_if(0) free_if(0))\
							out(mic_f2:length(mic_nbat->natoms*mic_nbat->fstride)alloc_if(0) free_if(0))wait(sig_mic2)\

							{
							}
						
		int j,part,fi=0;;
		part = mic_nbat->natoms*mic_nbat->fstride/(24-1);
		#pragma omp parallel for schedule(static)num_threads(24)
		for(j = 0;j < 24;j++){
			int i;
			if(j == 23){
				for(i = part*23;i < mic_nbat->natoms*mic_nbat->fstride;i ++){
					f[i] = f[i] + mic_f[i] + mic_f1[i] + mic_f2[i] + cpu_f[i];
				}
			}
			else{
				for(i = 0;i < part;i ++){
					f[part*j+i] = f[part*j+i] + mic_f[part*j+i] + mic_f1[part*j+i] + mic_f2[part*j+i] + cpu_f[part*j+i];
				}
			}
		}
		for(fi=0;fi<SHIFTS*DIM;fi++){
				fshift[fi] =fshift[fi] + mic_fshift[fi] + mic_fshift1[fi]  + mic_fshift2[fi]  + cpu_fshift[fi];
		}
		
		real *mic_tmp_Vc = Vc;
		real *mic_tmp_Vvdw = Vvdw;
		mic_tmp_Vc[0] = mic_tmp_Vc[0] + mic_Vc[0] + mic_Vc1[0] + mic_Vc2[0] + cpu_Vc[0];
		mic_tmp_Vvdw[0] = mic_tmp_Vvdw[0] + mic_Vvdw[0] + mic_Vvdw1[0] + mic_Vvdw2[0] + cpu_Vvdw[0];
		// for(j=0;j<mic_nbat->natoms*mic_nbat->fstride;j++){
			// if(cpu_f[j]-mic_f[j]>10.0){
				// printf("j:%d f:%f cpu:%f mic:%f + mic1:%f + mic2:%f = %f\n",j,f[j],cpu_f[j],mic_f[j],mic_f1[j],mic_f2[j],mic_f[j]+mic_f1[j]+mic_f2[j]);
			// }
		// }
	}
    // if (debug)
    // {
        // fprintf(debug, "number of half %dx%d atom pairs: %d after pruning: %d fraction %4.2f\n",
                // nbl->na_ci, nbl->na_ci,
                // nhwu, nhwu_pruned, nhwu_pruned/(double)nhwu);
        // fprintf(debug, "generic kernel pair interactions:            %d\n",
                // nhwu*nbl->na_ci/2*nbl->na_ci);
        // fprintf(debug, "generic kernel post-prune pair interactions: %d\n",
                // nhwu_pruned*nbl->na_ci/2*nbl->na_ci);
        // fprintf(debug, "generic kernel non-zero pair interactions:   %d\n",
                // npair_tot);
        // fprintf(debug, "ratio non-zero/post-prune pair interactions: %4.2f\n",
                // npair_tot/(double)(nhwu_pruned*nbl->na_ci/2*nbl->na_ci));
    // }
}
