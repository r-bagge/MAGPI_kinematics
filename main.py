import shutil
import numpy as np
from astropy.io import fits
import pandas as pd
import os
from kinemetry import kinemetry
import multiprocessing
import sys
from magpi_kinemetry import MAGPI_kinemetry
from kinemetry_plots import clean_images_velo
from kinemetry_plots import clean_images_flux
from kinemetry_plots import BPT_plots
from kinemetry_plots import stellar_gas_plots
from pafit.fit_kinematic_pa import fit_kinematic_pa

def monte_carlo(args):
    g_model, g_img, g_img_err, q_g, gas_kin_pa, x0_g, y0_g, rad_g, sg, vg, pa, n, catch, s_model, s_img, s_img_err, q_s, stellar_kin_pa, x0_s, y0_s, rad_s, ss, vs = args
    if catch == 1:
        v_asym_gmc = np.zeros(n)
        for h in range(n):
            model = g_img
            model += np.random.normal(loc=0, scale=g_img_err)
            k  = kinemetry(img=model, x0=x0_g, y0=y0_g, ntrm=11, plot=False, verbose=False, radius=rad_g,
                                bmodel=True, rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1],
                                allterms=True)
            k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
            k2 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k3 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            k4 = np.sqrt(k.cf[:, 6] ** 2 + k.cf[:, 7] ** 2)
            k5 = np.sqrt(k.cf[:, 8] ** 2 + k.cf[:, 10] ** 2)
            s05 = np.sqrt(0.5 * vg ** 2 + sg ** 2)
            v_asym = (k2 + k3 + k4 + k5) / (4 * s05)
            v_asym_gmc[h] = v_asym[-1]

        out = np.zeros(v_asym_gmc.shape)
        out[out == 0] = np.nan
        return v_asym_gmc, out
    elif catch == 2:
        v_asym_smc = np.zeros(n)
        for h in range(n):
            model = s_img
            model += np.random.normal(loc=0, scale=s_img_err)
            k = kinemetry(img=model, x0=x0_s, y0=y0_s, ntrm=11, plot=False, verbose=False, radius=rad_s,
                          bmodel=True, rangePA=[0, 360], rangeQ=[q_s - 0.1, q_s + 0.1],
                          allterms=True)
            k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
            k2 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k3 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            k4 = np.sqrt(k.cf[:, 6] ** 2 + k.cf[:, 7] ** 2)
            k5 = np.sqrt(k.cf[:, 8] ** 2 + k.cf[:, 10] ** 2)
            s05 = np.sqrt(0.5 * vs ** 2 + ss ** 2)
            v_asym = (k2 + k3 + k4 + k5) / (4 * s05)
            v_asym_smc[h] = v_asym[-1]

        out = np.zeros(v_asym_smc.shape)
        out[out == 0] = np.nan
        return out, v_asym_smc
    elif catch == 3:
        v_asym_smc = np.zeros(n)
        v_asym_gmc = np.zeros(n)
        for h in range(n):
            s_model_2 = s_img
            g_model_2 = g_img

            s_model_2 += np.random.normal(loc=0, scale=s_img_err)
            g_model_2 += np.random.normal(loc=0, scale=g_img_err)

            ks = kinemetry(img=s_model_2, x0=x0_s, y0=y0_s, ntrm=11, plot=False, verbose=False, radius=rad_s,
                          bmodel=True, rangePA=[0, 360], rangeQ=[q_s - 0.1, q_s + 0.1],
                          allterms=True)

            kg = kinemetry(img=g_model_2, x0=x0_g, y0=y0_g, ntrm=11, plot=False, verbose=False, radius=rad_g,
                          bmodel=True, rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1],
                          allterms=True)
            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
            ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)
            s05 = np.sqrt(0.5 * vs ** 2 + ss ** 2)
            v_asym_s = (ks2 + ks3 + ks4 + ks5) / (4 * s05)

            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
            kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
            kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
            kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)
            s05 = np.sqrt(0.5 * vg ** 2 + sg ** 2)
            v_asym_g = (kg2 + kg3 + kg4 + kg5) / (4 * s05)
            v_asym_gmc[h] = v_asym_g[-1]
            v_asym_smc[h] = v_asym_s[-1]

        return v_asym_gmc, v_asym_smc


def monte_carlo_parallel(pars):
    g_model, g_img, g_img_err, q_g, gas_kin_pa, x0_g, y0_g, rad_g, sg, vg,pa, n, catch, s_model, s_img, s_img_err, q_s, stellar_kin_pa, x0_s, y0_s, rad_s, ss, vs = pars
    cores = None
    if cores is None:
        cores = multiprocessing.cpu_count()
    print(f"Running {n} monte carlos on {cores} Cores!")
    group_size = n // 20
    args = [(g_model, g_img, g_img_err, q_g, gas_kin_pa, x0_g, y0_g, rad_g, sg, vg, pa, group_size, catch, s_model, s_img, s_img_err, q_s, stellar_kin_pa, x0_s, y0_s, rad_s, ss, vs) for _ in range(20)]
    ctx = multiprocessing.get_context()
    pool = ctx.Pool(processes=cores, maxtasksperchild=1)
    try:
        vasyms = pool.map(monte_carlo, args, chunksize=1)
    except KeyboardInterrupt:
        print("Caught kbd interrupt")
        pool.close()
        sys.exit(1)
    else:
        pool.close()
        pool.join()
        mcs = np.empty((2, n), dtype=np.float64)
        for i, r in enumerate(vasyms):
            start = i * group_size
            end = start + group_size
            mcs[:, start:end] = r

    return mcs


def MAGPI_kinemetry_parrallel(args):
    galaxy, pa, q, z, r50, quality = args
    field = str(galaxy)[:4]
    res_cutoff = 0.7 / 0.2
    cutoff = 1
    n_ells = 3
    n = 100
    SNR_Gas = 20
    SNR_Star = 3
    logfile = open("MAGPI_Plots/logfile.txt", "w")
    if z > 0.4:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!\n")
        return
    elif z < 0.2:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!\n")
        return
    if quality < 3:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift failed QC check!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift failed QC check!\n")
        return
    elif r50/res_cutoff < cutoff:
        print(f"MAGPIID = {galaxy}, r50 = {r50:.2f} pix, not resolved enough!")
        logfile.write(f"MAGPIID = {galaxy}, r50 = {r50:.2f} pix, not resolved enough!\n")
        return
    elif galaxy == int("1207128248") or galaxy == int("1506117050") or galaxy == int("1207197197"):
        print(f"MAGPIID = {galaxy}, fixing PA")
        logfile.write(f"MAGPIID = {galaxy}, fixing PA\n")
        pa = pa - 90
    elif galaxy == int("1501180123") or galaxy == int("1502293058") or galaxy == int("1203152196"):
        print(f"Piece of Shit")
        logfile.write(f"Piece of Shit\n")
        return
    else:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift passed!")
        print(f"MAGPIID = {galaxy}, r50 = {r50:.3f}, Res. passed!")
        print(f"MAGPIID = {galaxy} is {(r50 / res_cutoff):.3f} beam elements!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift passed!\n")
        logfile.write(f"MAGPIID = {galaxy}, r50 = {r50:.3f}, Res. passed!\n")
        logfile.write(f"MAGPIID = {galaxy} is {(r50 / res_cutoff):.3f} beam elements!\n")
    star_file = "MAGPI_Absorption_Line/kinematic_maps_latest/" + str(galaxy) + "_kinematics_ppxf-maps.fits"
    gas_file = "MAGPI_Emission_Line/MAGPI" + field + "/MAGPI" + field + "_v2.2.2_GIST_EmissionLine_Maps/MAGPI" + str(
        galaxy) + "_GIST_EmissionLines.fits"

    if os.path.exists(star_file):
        star_file_catch=True
    else:
        print("No stellar kinematics!")
        logfile.write("No stellar kinematics!\n")
        star_file_catch = False

    if os.path.exists(gas_file):
        gas_file_catch = True
    else:
        print("No gas kinematics!")
        logfile.write("No gas kinematics!\n")
        gas_file_catch = False

    # Check to see if there is neither gas or star data
    if star_file_catch == False and gas_file_catch == False:
        print("No kinematics! Skipping " + str(galaxy) + "!")
        logfile.write("No kinematics! Skipping " + str(galaxy) + "!\n")
        return

    # Gas kinemetry
    if star_file_catch == False and gas_file_catch:
        gasfile = fits.open(gas_file)
        g_flux, g_flux_err, g_velo, g_velo_err, g_sigma, g_sigma_err = gasfile["HA_F"].data, gasfile["HA_FERR"].data, \
            gasfile["V_GAS"].data, gasfile["VERR_GAS"].data, gasfile["SIGMA_GAS"].data, gasfile["SIGMAERR_GAS"].data
        gasfile.close()
        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_flux=g_flux / g_flux_err, limit=3)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_flux=g_flux / g_flux_err,limit=3)
        g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q, img_flux=g_flux / g_flux_err, limit=3)
        g_sigma_err = clean_images_velo(g_sigma_err, pa, r50, r50 * q, img_flux=g_flux / g_flux_err,limit=3)
        g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err

        clip = np.nanmax(g_flux)
        y0, x0 = g_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Gas SNR = {clip:.2f}...")
        print(f"Max Gas SNR = {clip:.2f}...", file=logfile)
        if clip < SNR_Gas:
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            logfile.write(
                "Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            logfile.write(
                "Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            return
        print("Doing kinemetry on gas only!")
        print("Doing kinemetry on gas only!", file=logfile)

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 2 * r50
        rad = np.arange(start, end, step)

        g_velo = g_velo - np.nanmedian(g_velo)
        ny, nx = g_velo.shape
        mx_img = np.max(g_velo.shape)
        x = (np.arange(0, nx))
        y = (np.arange(0, ny))
        xx, yy = np.meshgrid(x, y)
        xbin = xx.ravel() - np.median(xx.ravel())
        ybin = yy.ravel() - np.median(yy.ravel())
        gas_moment = g_velo.ravel()
        xbin = xbin[~np.isnan(gas_moment)]
        ybin = ybin[~np.isnan(gas_moment)]
        gas_moment = gas_moment[~np.isnan(gas_moment)]
        pa_g, pa_g_err, vsys = fit_kinematic_pa(xbin, ybin, gas_moment, plot=False, quiet=True, nsteps=50)

        kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_g, q]), allterms=True)
        vrotg = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
        q0 = 0.2
        q1 = np.sqrt((q ** 2 - q0 ** 2) / (1 - q0 ** 2))
        if q1 < 0.2 or np.isnan(q1):
            q1 = 0.2
        vrotg = np.nanmax(vrotg) / np.sin(np.arccos(q1))

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)

        kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa-90, q]), even=True)

        kg_sigma = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True, paq=np.array([pa-90, q]), even=True)
        sg = np.nansum(kg_flux.cf[:, 0] * kg_sigma.cf[:, 0])/np.nansum(kg_flux.cf[:, 0])
        kg_velo = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_g, q]),
                       allterms=True)

        return [kg_velo.velkin, g_velo, g_velo_err, q, pa, x0, y0, rad, sg, vrotg, pa, n, 1, None, None, None, None, None, None, None, None, None, None]

    # Stellar kinemetry
    if star_file_catch and gas_file_catch == False:
        starfile = fits.open(star_file)
        s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
        starfile.close()

        s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_flux=s_flux,limit=3)
        s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_flux=s_flux,limit=3)
        s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_flux=s_flux,limit=3)

        clip = np.nanmax(s_flux)
        y0, x0 = s_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Stellar SNR = {clip:.2f}...")
        print(f"Max Stellar SNR = {clip:.2f}...", file=logfile)
        if clip < SNR_Star:
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            logfile.write(
                "Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            logfile.write(
                "Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            return
        print("Doing kinemetry on stars only!")
        print("Doing kinemetry on stars only!", file=logfile)

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 2 * r50
        rad = np.arange(start, end, step)

        s_velo = s_velo - np.nanmedian(s_velo)
        ny, nx = s_velo.shape
        mx_img = np.max(s_velo.shape)
        x = (np.arange(0, nx))
        y = (np.arange(0, ny))
        xx, yy = np.meshgrid(x, y)
        xbin = xx.ravel() - np.median(xx.ravel())
        ybin = yy.ravel() - np.median(yy.ravel())
        star_moment = s_velo.ravel()
        xbin = xbin[~np.isnan(star_moment)]
        ybin = ybin[~np.isnan(star_moment)]
        star_moment = star_moment[~np.isnan(star_moment)]
        pa_s, pa_s_err, vsys = fit_kinematic_pa(xbin, ybin, star_moment, plot=False, quiet=True, nsteps=50)

        ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_s, q]), allterms=True)

        vrots = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
        q0 = 0.2
        q1 = np.sqrt((q ** 2 - q0 ** 2) / (1 - q0 ** 2))
        if q1 < 0.2 or np.isnan(q1):
            q1 = 0.2
        vrots = np.nanmax(vrots) / np.sin(np.arccos(q1))

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)

        ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa-90, q]), even=True)

        ks_sigma = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True, paq=np.array([pa-90, q]), even=True)

        ss = np.nansum(ks_flux.cf[:, 0] * ks_sigma.cf[:, 0])/np.nansum(ks_flux.cf[:, 0])
        ks_velo = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_s, q]),
                       allterms=True)

        return [None, None, None, None, None, None, None, None, None, None, pa, n, 2, ks_velo.velkin, s_velo, s_velo_err, q, pa, x0, y0, rad, ss, vrots]

    if star_file_catch and gas_file_catch:
        starfile = fits.open(star_file)
        gasfile = fits.open(gas_file)
        s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
        starfile.close()

        s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_flux=s_flux,limit=3)
        s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_flux=s_flux,limit=3)
        s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_flux=s_flux, limit=3)

        g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data, gasfile[11].data
        gasfile.close()

        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_flux=g_flux / g_flux_err,limit=3)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_flux= g_flux / g_flux_err,limit=3)
        g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err

        s_clip = np.nanmax(s_flux)
        y0, x0 = s_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)

        print(f"Max Stellar SNR = {s_clip:.2f}...")
        logfile.write(f"Max Stellar SNR = {s_clip:.2f}...\n")
        if s_clip < SNR_Star or np.isinf(SNR_Star) or np.isnan(SNR_Star):
            print("Not doing kinemetry on " + str(galaxy) + " because its stars are heinous looking")
            logfile.write(
                "Not doing kinemetry on " + str(galaxy) + " because its stars are heinous looking\n")
            print("Trying the gas...")
            print("Trying the gas...\n", file=logfile)
            g_clip = np.nanmax(g_flux)
            print(f"Max Gas SNR = {g_clip:.2f}...")
            logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
            if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                print(
                    "Not doing kinemetry on " + str(
                        galaxy) + "because its gas is also heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(
                        galaxy) + " because its gas is also heinous looking\n")
                return
            else:
                print("Doing kinemetry on gas only!")
                print("Doing kinemetry on gas only!", file=logfile)

                step = (0.65 / 2) / 0.2
                start = (0.65 / 2) / 0.2 - step
                end = 2 * r50
                rad = np.arange(start, end, step)

                g_velo = g_velo - np.nanmedian(g_velo)
                ny, nx = g_velo.shape
                mx_img = np.max(g_velo.shape)
                x = (np.arange(0, nx))
                y = (np.arange(0, ny))
                xx, yy = np.meshgrid(x, y)
                xbin = xx.ravel() - np.median(xx.ravel())
                ybin = yy.ravel() - np.median(yy.ravel())
                gas_moment = g_velo.ravel()
                xbin = xbin[~np.isnan(gas_moment)]
                ybin = ybin[~np.isnan(gas_moment)]
                gas_moment = gas_moment[~np.isnan(gas_moment)]
                pa_g, pa_g_err, vsys = fit_kinematic_pa(xbin, ybin, gas_moment, plot=False, quiet=True, nsteps=50)

                kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, paq=np.array([pa_g, q]), allterms=True)
                vrotg = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                q0 = 0.2
                q1 = np.sqrt((q ** 2 - q0 ** 2) / (1 - q0 ** 2))
                if q1 < 0.2 or np.isnan(q1):
                    q1 = 0.2
                vrotg = np.nanmax(vrotg) / np.sin(np.arccos(q1))

                step = (0.65 / 2) / 0.2
                start = (0.65 / 2) / 0.2 - step
                end = 1 * r50
                rad = np.arange(start, end, step)

                kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                    bmodel=True, paq=np.array([pa-90, q]), even=True)

                kg_sigma = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True, paq=np.array([pa-90, q]), even=True)
                sg = np.nansum(kg_flux.cf[:, 0] * kg_sigma.cf[:, 0])/np.nansum(kg_flux.cf[:, 0])

                kg_velo = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, paq=np.array([pa_g, q]),
                               allterms=True)

                return [kg_velo.velkin, g_velo, g_velo_err, q, pa, x0, y0, rad, sg, vrotg, pa, n, 1, None, None, None, None, None, None, None, None, None, None]

        g_clip = np.nanmax(g_flux)
        print(f"Max Gas SNR = {g_clip:.2f}...")
        logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
        if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its gas is heinous looking")
            logfile.write(
                "Not Plotting or doing Kinemetry on " + str(galaxy) + " because its gas is heinous looking\n")
            print("Trying the stars...")
            print("Trying the stars...", file=logfile)
            s_clip = np.nanmax(s_flux)
            print(f"Max Star SNR = {s_clip:.2f}...")
            logfile.write(f"Max Star SNR = {s_clip:.2f}...\n")
            if s_clip < SNR_Star or np.isinf(s_clip) or np.isnan(s_clip):
                print(
                    "Not doing kinemetry on " + str(galaxy) + "because its gas are also heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(
                        galaxy) + " because its gas are also heinous looking\n")
                return
            else:
                step = (0.65 / 2) / 0.2
                start = (0.65 / 2) / 0.2 - step
                end = 1 * r50
                rad = np.arange(start, end, step)
                if len(rad) < n_ells:
                    print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                    logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                    return

                print("Doing kinemetry on stars only!")
                print("Doing kinemetry on stars only!", file=logfile)

                step = (0.65 / 2) / 0.2
                start = (0.65 / 2) / 0.2 - step
                end = 2 * r50
                rad = np.arange(start, end, step)

                s_velo = s_velo - np.nanmedian(s_velo)
                ny, nx = s_velo.shape
                mx_img = np.max(g_velo.shape)
                x = (np.arange(0, nx))
                y = (np.arange(0, ny))
                xx, yy = np.meshgrid(x, y)
                xbin = xx.ravel() - np.median(xx.ravel())
                ybin = yy.ravel() - np.median(yy.ravel())
                star_moment = s_velo.ravel()
                xbin = xbin[~np.isnan(star_moment)]
                ybin = ybin[~np.isnan(star_moment)]
                star_moment = star_moment[~np.isnan(star_moment)]
                pa_s, pa_s_err, vsys = fit_kinematic_pa(xbin, ybin, star_moment, plot=False, quiet=True, nsteps=50)

                ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, paq=np.array([pa_s, q]), allterms=True)

                vrots = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                q0 = 0.2
                q1 = np.sqrt((q ** 2 - q0 ** 2) / (1 - q0 ** 2))
                if q1 < 0.2 or np.isnan(q1):
                    q1 = 0.2
                vrots = np.nanmax(vrots) / np.sin(np.arccos(q1))

                step = (0.65 / 2) / 0.2
                start = (0.65 / 2) / 0.2 - step
                end = 1 * r50
                rad = np.arange(start, end, step)

                ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                    bmodel=True, paq=np.array([pa-90, q]), even=True)
                ks_sigma = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True, paq=np.array([pa-90, q]), even=True)
                ss = np.nansum(ks_flux.cf[:, 0] * ks_sigma.cf[:, 0])/np.nansum(ks_flux.cf[:, 0])

                ks_velo = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, paq=np.array([pa_s, q]),
                               allterms=True)

                return [None, None, None, None, None, None, None, None, None, None, pa, n, 2, ks_velo.velkin, s_velo, s_velo_err, q, pa, x0, y0, rad, ss, vrots]

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return

        print("Doing kinemetry on stars and gas!")
        print("Doing kinemetry on stars and gas!", file=logfile)

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 2 * r50
        rad = np.arange(start, end, step)

        g_velo = g_velo - np.nanmedian(g_velo)
        ny, nx = g_velo.shape
        mx_img = np.max(g_velo.shape)
        x = (np.arange(0, nx))
        y = (np.arange(0, ny))
        xx, yy = np.meshgrid(x, y)
        xbin = xx.ravel() - np.median(xx.ravel())
        ybin = yy.ravel() - np.median(yy.ravel())
        gas_moment = g_velo.ravel()
        xbin = xbin[~np.isnan(gas_moment)]
        ybin = ybin[~np.isnan(gas_moment)]
        gas_moment = gas_moment[~np.isnan(gas_moment)]
        pa_g, pa_g_err, vsys = fit_kinematic_pa(xbin, ybin, gas_moment, plot=False, quiet=True, nsteps=50)

        s_velo = s_velo - np.nanmedian(s_velo)
        ny, nx = s_velo.shape
        mx_img = np.max(g_velo.shape)
        x = (np.arange(0, nx))
        y = (np.arange(0, ny))
        xx, yy = np.meshgrid(x, y)
        xbin = xx.ravel() - np.median(xx.ravel())
        ybin = yy.ravel() - np.median(yy.ravel())
        star_moment = s_velo.ravel()
        xbin = xbin[~np.isnan(star_moment)]
        ybin = ybin[~np.isnan(star_moment)]
        star_moment = star_moment[~np.isnan(star_moment)]
        pa_s, pa_s_err, vsys = fit_kinematic_pa(xbin, ybin, star_moment, plot=False, quiet=True, nsteps=50)

        kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_g, q]), allterms=True)
        vrotg = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
        q0 = 0.2
        q1 = np.sqrt((q ** 2 - q0 ** 2) / (1 - q0 ** 2))
        if q1 < 0.2 or np.isnan(q1):
            q1 = 0.2
        vrotg = np.nanmax(vrotg) / np.sin(np.arccos(q1))

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)

        kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa-90, q]), even=True)

        kg_sigma = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True, paq=np.array([pa-90, q]), even=True)
        sg = np.nansum(kg_flux.cf[:, 0] * kg_sigma.cf[:, 0])/np.nansum(kg_flux.cf[:, 0])
        kg_velo = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_g, q]),
                       allterms=True)

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 2 * r50
        rad = np.arange(start, end, step)
        ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_s, q]), allterms=True)

        vrots = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
        q0 = 0.2
        q1 = np.sqrt((q ** 2 - q0 ** 2) / (1 - q0 ** 2))
        if q1 < 0.2 or np.isnan(q1):
            q1 = 0.2
        vrots = np.nanmax(vrots) / np.sin(np.arccos(q1))

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50
        rad = np.arange(start, end, step)

        ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa-90, q]), even=True)
        ks_sigma = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True, paq=np.array([pa-90, q]), even=True)
        ss = np.nansum(ks_flux.cf[:, 0] * ks_sigma.cf[:, 0])/np.nansum(ks_flux.cf[:, 0])

        ks_velo = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, paq=np.array([pa_s, q]),
                       allterms=True)

        return [kg_velo.velkin, g_velo, g_velo_err, q, pa, x0, y0, rad, sg, vrotg, pa, n, 3, ks_velo.velkin, s_velo, s_velo_err, q, pa, x0, y0, rad, ss, vrots]


if __name__ == '__main__':
    mc = True
    if mc == True:
        file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=17)
        print(file["z"])
        z = file["z"].to_numpy()
        pa = file["ang_it"].to_numpy()
        q = file["axrat_it"].to_numpy()
        re = file["R50_it"].to_numpy() / 0.2
        quality = file["QOP"].to_numpy()
        galaxy = file["MAGPIID"].to_numpy()
        galaxies = []
        GasAsym = []
        GasAsymErr = []
        StarsAsym = []
        StarsAsymErr = []
        print("Beginning the hard part...")
        for i in range(len(file)):
            pars = [galaxy[i], pa[i], q[i], z[i], re[i], quality[i]]
            args = MAGPI_kinemetry_parrallel(pars)
            if args is None:
                continue
            mcs = monte_carlo_parallel(args)
            galaxies.append(galaxy[i])
            print(f"Gas Asym={np.nanmean(mcs[0]):.2f}")
            GasAsym.append(np.nanmean(mcs[0]))
            GasAsymErr.append(np.nanstd(mcs[0]))
            print(f"Stars Asym={np.nanmean(mcs[1]):.2f}")
            StarsAsym.append(np.nanmean(mcs[1]))
            StarsAsymErr.append(np.nanstd(mcs[1]))

        print("Beginning the easy part...")
        if os.path.exists("MAGPI_Plots/plots/Gas_Stars_Ellipse"):
            shutil.rmtree("MAGPI_Plots/plots/Gas_Stars_Ellipse")
            os.mkdir("MAGPI_Plots/plots/Gas_Stars_Ellipse")
        if os.path.exists("MAGPI_Plots/plots/rotation_curves"):
            shutil.rmtree("MAGPI_Plots/plots/rotation_curves")
            os.mkdir("MAGPI_Plots/plots/rotation_curves")
        results = MAGPI_kinemetry(source_cat="MAGPI_csv/MAGPI_master_source_catalogue.csv",sample=galaxies,
                                  n_ells=3, SNR_Star=3, SNR_Gas=20)
        print("Beginning the second easy part...")
        if os.path.exists("MAGPI_Plots/plots/flux_velo_plots"):
            shutil.rmtree("MAGPI_Plots/plots/flux_velo_plots")
            os.mkdir("MAGPI_Plots/plots/flux_velo_plots")

        df = pd.DataFrame({"MAGPIID": galaxies,
                           "z":results[1],
                           "v_asym_g": GasAsym,
                           "v_asym_g_err": GasAsymErr,
                           "v_asym_s": StarsAsym,
                           "v_asym_s_err": StarsAsymErr,
                           "PA_g": results[2],
                           "PA_s": results[3],
                           "D_PA": results[4],
                           "V_rot_g": results[5],
                           "V_rot_s": results[6],
                           "Sigma_g": results[7],
                           "Sigma_s": results[8],
                           "SNR_g": results[9],
                           "SNR_s": results[10],
                           "re_over_psf":results[13]
                           })
        df = df[~df["MAGPIID"].isin(df[(np.isnan(df.v_asym_s)) & (np.isnan(df.v_asym_g))]["MAGPIID"])]
        df.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_s05.csv",index=False)
        # df.to_csv("/Users/z5408076/Documents/OneDrive - UNSW/MAGPI_csv/MAGPI_kinemetry_sample_s05.csv",index=False)
        file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
        file1 = file[file["MAGPIID"].isin(df.MAGPIID)]
        file1.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_source_catalogue.csv", index=False)
        print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
        if os.path.exists("MAGPI_Plots/plots/fitsfiles"):
            shutil.rmtree("MAGPI_Plots/plots/fitsfiles")
            os.mkdir("MAGPI_Plots/plots/fitsfiles")
        if os.path.exists("MAGPI_Plots/plots/kinemetry_model_plots"):
            shutil.rmtree("MAGPI_Plots/plots/kinemetry_model_plots")
            os.mkdir("MAGPI_Plots/plots/kinemetry_model_plots")
        # stellar_gas_plots_vectorized = np.vectorize(stellar_gas_plots)
        # stellar_gas_plots_vectorized(results[0])
        BPT_plots("MAGPI_csv/MAGPI_kinemetry_sample_s05_BPT.csv", "MAGPI_csv/MAGPI_kinemetry_sample_s05.csv", n_re=1.0)

    else:
        print("Beginning the easy part...")
        if os.path.exists("MAGPI_Plots/plots/Gas_Stars_Ellipse"):
            shutil.rmtree("MAGPI_Plots/plots/Gas_Stars_Ellipse")
            os.mkdir("MAGPI_Plots/plots/Gas_Stars_Ellipse")
        results = MAGPI_kinemetry(source_cat="MAGPI_csv/MAGPI_master_source_catalogue.csv",
                                  n_ells=3, SNR_Star=3, SNR_Gas=20)
        print("Beginning the second easy part...")
        if os.path.exists("MAGPI_Plots/plots/flux_velo_plots"):
            shutil.rmtree("MAGPI_Plots/plots/flux_velo_plots")
            os.mkdir("MAGPI_Plots/plots/flux_velo_plots")

        df = pd.DataFrame({"MAGPIID":results[0],
                           "z": results[1],
                           "v_asym_g":results[11],
                           "v_asym_g_err":np.zeros(len(results[10])),
                           "v_asym_s":results[12],
                           "v_asym_s_err":np.zeros(len(results[11])),
                           "PA_g": results[2],
                           "PA_s": results[3],
                           "D_PA": results[4],
                           "V_rot_g": results[5],
                           "V_rot_s": results[6],
                           "Sigma_g": results[7],
                           "Sigma_s": results[8],
                           "SNR_g": results[9],
                           "SNR_s": results[10],
                           "re_over_psf": results[13]
                           })
        df = df[~df["MAGPIID"].isin(df[(np.isnan(df.v_asym_s)) & (np.isnan(df.v_asym_g))]["MAGPIID"])]
        df.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_s05_no_err.csv", index=False)
        file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
        file1 = file[file["MAGPIID"].isin(df.MAGPIID)]
        file1.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_source_catalogue.csv", index=False)

        print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
        stellar_gas_plots_vectorized = np.vectorize(stellar_gas_plots)
        stellar_gas_plots_vectorized(results[0])
        BPT_plots("MAGPI_csv/MAGPI_kinemetry_sample_s05_BPT.csv", "MAGPI_csv/MAGPI_kinemetry_sample_s05.csv", n_re=1.0)
