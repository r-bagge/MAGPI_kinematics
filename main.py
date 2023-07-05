import numpy as np
from astropy.io import fits
import pandas as pd
import os
from kinemetry import kinemetry
import multiprocessing
import sys
from magpi_kinemetry import MAGPI_kinemetry
from magpi_kinemetry import clean_images

def monte_carlo(args):
    g_model, g_img_err, q_g, x0_g, y0_g, rad_g, vg_rot, vg_sig, kg_flux_k0, n, catch, s_model, s_img_err, q_s, x0_s, y0_s, rad_s, vs_rot, vs_sig, ks_flux_k0 = args
    if catch == 1:
        v_asym_gmc = np.zeros(n)
        for h in range(n):
            model = g_model
            model += np.random.normal(loc=g_model, scale=g_img_err)
            k = kinemetry(img=model, x0=x0_g, y0=y0_g, ntrm=11, plot=False, verbose=False, radius=rad_g, bmodel=True,
                          rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1], allterms=True, cover=0.95)

            k2 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k3 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            k4 = np.sqrt(k.cf[:, 6] ** 2 + k.cf[:, 7] ** 2)
            k5 = np.sqrt(k.cf[:, 8] ** 2 + k.cf[:, 10] ** 2)

            kgs0 = np.nanmean(vg_sig[(rad_g/np.median(rad_g)) < 1])
            inc = np.arccos(np.sqrt(q_g ** 2 - 0.2 ** 2) / (1 - 0.2 ** 2))
            vg_rot = vg_rot / (2 * np.sin(inc))
            gs05 = np.sqrt(0.5 * np.nanmax(vg_rot) ** 2 + kgs0 ** 2)
            vasym_g = k2 + k3 + k4 + k5
            vasym_g = vasym_g / (4 * gs05)
            vasym_g[np.isnan(vasym_g)] = 0
            v_asym_gmc[h] = np.median(vasym_g)

        out = np.zeros(v_asym_gmc.shape)
        out[out==0]=np.nan
        return v_asym_gmc, out
    elif catch == 2:
        v_asym_smc = np.zeros(n)
        for h in range(n):
            model= s_model
            model += np.random.normal(loc=s_model, scale=s_img_err)
            k = kinemetry(img=model, x0=x0_s, y0=y0_s, ntrm=11, plot=False, verbose=False, radius=rad_s, bmodel=True,
                          rangePA=[0, 360], rangeQ=[q_s - 0.1, q_s + 0.1], allterms=True, cover=0.95)

            k2 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k3 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            k4 = np.sqrt(k.cf[:, 6] ** 2 + k.cf[:, 7] ** 2)
            k5 = np.sqrt(k.cf[:, 8] ** 2 + k.cf[:, 10] ** 2)
            kss0 = np.nanmean(vs_sig[(rad_s/np.median(rad_s)) < 1])
            inc = np.arccos(np.sqrt(q_s ** 2 - 0.2 ** 2) / (1 - 0.2 ** 2))
            vs_rot = vs_rot / (2 * np.sin(inc))
            ss05 = np.sqrt(0.5 * np.nanmax(vs_rot) ** 2 + kss0 ** 2)
            vasym_s = k2 + k3 + k4 + k5
            vasym_s = vasym_s / (4 * ss05)
            vasym_s[np.isnan(vasym_s)] = 0
            v_asym_smc[h] = np.median(vasym_s)
        out = np.zeros(v_asym_smc.shape)
        out[out == 0] = np.nan
        return out, v_asym_smc
    elif catch == 3:
        v_asym_smc = np.zeros(n)
        v_asym_gmc = np.zeros(n)
        for h in range(n):
            s_model_2 = s_model
            g_model_2 = g_model

            s_model_2 += np.random.normal(loc=s_model, scale=s_img_err)
            g_model_2 += np.random.normal(loc=g_model, scale=g_img_err)

            ks = kinemetry(img=s_model_2, x0=x0_s, y0=y0_s, ntrm=11, plot=False, verbose=False, radius=rad_s, bmodel=True,
                           rangePA=[0, 360], rangeQ=[q_s - 0.1, q_s + 0.1], allterms=True, cover=0.95)
            kg = kinemetry(img=g_model_2, x0=x0_g, y0=y0_g, ntrm=11, plot=False, verbose=False, radius=rad_g, bmodel=True,
                           rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1], allterms=True, cover=0.95)

            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
            ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)
            kss0 = np.nanmean(vs_sig[(rad_s/np.median(rad_s)) < 1])
            inc = np.arccos(np.sqrt(q_s ** 2 - 0.2 ** 2) / (1 - 0.2 ** 2))
            vs_rot = vs_rot / (2 * np.sin(inc))
            ss05 = np.sqrt(0.5 * np.nanmax(vs_rot) ** 2 + kss0 ** 2)
            vasym_s = ks2 + ks3 + ks4 + ks5
            vasym_s = vasym_s / (4 * ss05)
            vasym_s[np.isnan(vasym_s)] = 0
            v_asym_smc[h] = np.median(vasym_s)

            kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
            kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
            kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
            kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

            kgs0 = np.nanmean(vg_sig[(rad_g/np.median(rad_g)) < 1])
            inc = np.arccos(np.sqrt(q_s ** 2 - 0.2 ** 2) / (1 - 0.2 ** 2))
            vs_rot = vs_rot / (2 * np.sin(inc))
            gs05 = np.sqrt(0.5 * np.nanmax(vs_rot) ** 2 + kgs0 ** 2)
            vasym_g = kg2 + kg3 + kg4 + kg5
            vasym_g = vasym_g / (4 * gs05)
            vasym_g[np.isnan(vasym_g)] = 0
            v_asym_gmc[h] = np.median(vasym_g)
        return v_asym_gmc, v_asym_smc


def monte_carlo_parallel(pars):
    g_model, g_velo_err, q_g, x0_g, y0_g, rad_g, vg_rot, sg_sig, flux_g, n, catch, s_model, s_velo_err, q_s, x0_s, y0_s, rad_s, vs_rot, ss_sig, flux_s = pars
    cores = None
    if cores is None:
        cores = multiprocessing.cpu_count()
    print(f"Running {cores} Cores!")
    group_size = n // 20
    args = [(g_model, g_velo_err, q_g, x0_g, y0_g, rad_g, vg_rot, sg_sig, flux_g, group_size, catch, s_model, s_velo_err, q_s, x0_s, y0_s, rad_s, vs_rot, ss_sig, flux_s) for _ in range(20)]
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
    n_re = 2
    res_cutoff = 0.7 / 0.2
    cutoff = 1
    n_ells = 5
    n = 100
    SNR_Gas = 20
    SNR_Star = 3
    logfile = open("/Volumes/LDS/Astro/PhD/MAGPI/MAGPI_Maps/MAGPI" + field + "/MAGPI" + field + "_logfile.txt", "w")
    if z > 0.35:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!")
        #logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!\n")
        return
    elif z < 0.28:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!")
        #logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!\n")
        return
    elif quality < 3:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift failed QC check!")
        #logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift failed QC check!\n")
        return
    elif r50 < cutoff * res_cutoff:
        print(f"MAGPIID = {galaxy}, r50 = {r50:.2f} pix, not resolved enough!")
        #logfile.write(f"MAGPIID = {galaxy}, r50 = {r50:.2f} pix, not resolved enough!\n")
        return
    elif galaxy == int("1207128248") or galaxy == int("1506117050"):
        print(f"MAGPIID = {galaxy}, fixing PA")
        #logfile.write(f"MAGPIID = {galaxy}, fixing PA\n")
        pa = pa - 90
    elif galaxy == int("1207197197"):
        print(f"MAGPIID = {galaxy}, fixing PA")
        #logfile.write(f"MAGPIID = {galaxy}, fixing PA\n")
        pa = pa - 180
    elif galaxy == int("1501180123") or int("1502293058"):
        print(f"Piece of Shit")
        # logfile.write(f"MAGPIID = {galaxy[f]}, For Qainhui\n")
        return
    else:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift passed!")
        print(f"MAGPIID = {galaxy}, r50 = {r50:.3f}, Res. passed!")
        print(f"MAGPIID = {galaxy} is {(r50 / res_cutoff):.3f} beam elements!")
        #logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift passed!\n")
        #logfile.write(f"MAGPIID = {galaxy}, r50 = {r50:.3f}, Res. passed!\n")
        #logfile.write(f"MAGPIID = {galaxy} is {(r50 / res_cutoff):.3f} beam elements!\n")
    star_file = "/Volumes/LDS/Astro/PhD/MAGPI/MAGPI_Maps/MAGPI" + field + "/Absorption_Line/" + str(
        galaxy) + "_kinematics_ppxf-maps.fits"
    gas_file = "/Volumes/LDS/Astro/PhD/MAGPI/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
        galaxy) + "_GIST_EmissionLines.fits"

    if os.path.exists(star_file):
        star_file_catch = True
    else:
        print("No stellar kinematics!")
        #logfile.write("No stellar kinematics!\n")
        star_file_catch = False

    if os.path.exists(gas_file):
        gas_file_catch = True
    else:
        print("No gas kinematics!")
        #logfile.write("No gas kinematics!\n")
        gas_file_catch = False

    # Check to see if there is neither gas or star data
    if star_file_catch == False and gas_file_catch == False:
        print("No kinematics! Skipping " + str(galaxy) + "!")
        #logfile.write("No kinematics! Skipping " + str(galaxy) + "!\n")
        return

    # Gas kinemetry
    if star_file_catch == False and gas_file_catch:
        gasfile = fits.open(gas_file)
        g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                                                          gasfile[10].data, gasfile[11].data
        gasfile.close()
        g_velo = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = clean_images(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err

        clip = np.nanmax(g_flux)
        y0, x0 = g_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Gas SNR = {clip:.2f}...")
        print(f"Max Gas SNR = {clip:.2f}...", file=logfile)
        if clip < SNR_Gas:
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            #logfile.write("Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            #logfile.write("Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            #logfile.write.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return
        print("Doing kinemetry on gas only!")
        print("Doing kinemetry on gas only!", file=logfile)
        g_velo[np.isnan(g_velo)] = 0
        g_sigma[np.isnan(g_sigma)] = 0
        g_velo_err[np.isnan(g_velo_err)] = 0
        g_flux[np.isnan(g_flux)] = 0

        v_g = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                       cover=0.95)
        k_flux_g = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True,
                             rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                             cover=0.95)
        s_g = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True,
                             rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                             cover=0.95)
        flux_g = k_flux_g.cf[:,0]
        vg_rot = np.sqrt(v_g.cf[:,1]**2 + v_g.cf[:,2]**2)
        sg_sig = s_g.cf[:,0]

        return v_g.velkin, g_velo_err, q, x0, y0, rad, vg_rot, sg_sig, flux_g, n, 1, None, None, None, None, None, None, None, None, None

    # Stellar kinemetry
    if star_file_catch and gas_file_catch == False:
        starfile = fits.open(star_file)
        s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
        starfile.close()

        clip = np.nanmax(s_flux)
        y0, x0 = s_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Stellar SNR = {clip:.2f}...")
        print(f"Max Stellar SNR = {clip:.2f}...", file=logfile)
        if clip < SNR_Star:
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            #logfile.write("Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not doing kinemetry on " + str(galaxy) + " because its heinous looking")
            #logfile.write("Not doing kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return
        print("Doing kinemetry on stars only!")
        print("Doing kinemetry on stars only!", file=logfile)
        s_velo[np.isnan(s_velo)] = 0
        s_sigma[np.isnan(s_sigma)] = 0
        s_velo_err[np.isnan(s_velo_err)] = 0
        s_flux[np.isnan(s_flux)] = 0

        v_s = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                       cover=0.95)
        k_flux_s = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True,
                             rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                             cover=0.95)
        s_s = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True,
                             rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                             cover=0.95)
        flux_s = k_flux_s.cf[:, 0]
        v_s_rot = np.sqrt(v_s.cf[:,1]**2 + v_s.cf[:,2]**2)
        s_s_sig = s_s.cf[:,0]
        return None, None, None, None, None, None, None, None, None, n, 2, v_s.velkin, s_velo_err, q, x0, y0, rad, v_s_rot, s_s_sig, flux_s

    if star_file_catch and gas_file_catch:
        starfile = fits.open(star_file)
        gasfile = fits.open(gas_file)
        s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
        starfile.close()
        g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                                                          gasfile[10].data, gasfile[11].data
        gasfile.close()
        g_velo = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = clean_images(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err

        s_clip = np.nanmax(s_flux)
        y0, x0 = s_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)

        print(f"Max Stellar SNR = {s_clip:.2f}...")
        #logfile.write(f"Max Stellar SNR = {s_clip:.2f}...\n")
        if s_clip < SNR_Star or np.isinf(SNR_Star) or np.isnan(SNR_Star):
            print("Not doing kinemetry on " + str(galaxy) + " because its stars are heinous looking")
            #logfile.write.write("Not doing kinemetry on " + str(galaxy) + " because its stars are heinous looking\n")
            print("Trying the gas...")
            print("Trying the gas...\n", file=logfile)
            g_clip = np.nanmax(g_flux)
            print(f"Max Gas SNR = {g_clip:.2f}...")
            #logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
            if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                print(
                    "Not doing kinemetry on " + str(
                        galaxy) + "because its gas is also heinous looking")
                #logfile.write("Not doing kinemetry on " + str(galaxy) + " because its gas is also heinous looking\n")
                return
            else:
                print("Doing kinemetry on the gas only!")
                print("Doing kinemetry on the gas only!", file=logfile)
                start = (0.65 / 2) / 0.2
                step = (0.65 / 2) / 0.2
                end = n_re * r50
                rad = np.arange(start, end, step)
                if len(rad) < n_ells:
                    print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                    #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                    return
                print("Doing kinemetry on gas!")
                print("Doing kinemetry on gas!", file=logfile)
                g_velo[np.isnan(g_velo)] = 0
                g_sigma[np.isnan(g_sigma)] = 0
                g_velo_err[np.isnan(g_velo_err)] = 0
                g_flux[np.isnan(g_flux)] = 0

                v_g = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                                cover=0.95)
                k_flux_g = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                                     cover=0.95)
                s_g = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True,
                                rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                                cover=0.95)
                flux_g = k_flux_g.cf[:, 0]
                vg_rot = np.sqrt(v_g[:,1]**2 + v_g.cf[:, 2]**2)
                sg_sig = s_g.cf[:, 0]

                return v_g.velkin, g_velo_err, q, x0, y0, rad, vg_rot, sg_sig, flux_g, n, 1, None, None, None, None, None, None, None, None, None

        g_clip = np.nanmax(g_flux)
        print(f"Max Gas SNR = {g_clip:.2f}...")
        #logfile.write.write(f"Max Gas SNR = {g_clip:.2f}...\n")
        if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its gas is heinous looking")
            #logfile.write.write("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its gas is heinous looking\n")
            print("Trying the stars...")
            print("Trying the stars...", file=logfile)
            s_clip = np.nanmax(s_flux)
            print(f"Max Star SNR = {s_clip:.2f}...")
            #logfile.write(f"Max Star SNR = {s_clip:.2f}...\n")
            if s_clip < SNR_Star or np.isinf(s_clip) or np.isnan(s_clip):
                print(
                    "Not doing kinemetry on " + str(galaxy) + "because its gas are also heinous looking")
                #logfile.write("Not doing kinemetry on " + str(galaxy) + " because its stars are also heinous looking\n")
                return
            else:
                start = (0.65 / 2) / 0.2
                step = (0.65 / 2) / 0.2
                end = n_re * r50
                rad = np.arange(start, end, step)
                if len(rad) < n_ells:
                    print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                    #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                    return
                print("Doing kinemetry on stars only!")
                print("Doing kinemetry on stars only!", file=logfile)
                s_velo[np.isnan(s_velo)] = 0
                s_sigma[np.isnan(s_sigma)] = 0
                s_velo_err[np.isnan(s_velo_err)] = 0
                s_flux[np.isnan(s_flux)] = 0

                v_s = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                                cover=0.95)
                k_flux_s = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                                     cover=0.95)
                s_s = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True,
                                rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                                cover=0.95)
                flux_s = k_flux_s.cf[:, 0]
                v_s_rot = np.sqrt(v_s.cf[:,1]**2 + v_s.cf[:, 2]**2)
                s_s_sig = s_s.cf[:, 0]

                return None, None, None, None, None, None, None, None, None, n, 2, v_s.velkin, s_velo_err, q, x0, y0, rad, v_s_rot, s_s_sig, flux_s

        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return
        s_velo[np.isnan(s_velo)] = 0
        s_sigma[np.isnan(s_sigma)] = 0
        g_velo[np.isnan(g_velo)] = 0
        g_sigma[np.isnan(g_sigma)] = 0
        g_velo_err[np.isnan(g_velo_err)] = 0
        g_flux[np.isnan(g_flux)] = 0
        s_flux[np.isnan(s_flux)] = 0

        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return

        print("Doing kinemetry on stars and gas!")
        print("Doing kinemetry on stars and gas!", file=logfile)
        v_s = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                        bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                        cover=0.95)
        k_flux_s = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True,
                             rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                             cover=0.95)
        s_s = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                        bmodel=True,
                        rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                        cover=0.95)
        flux_s = k_flux_s.cf[:, 0]
        v_s_rot = np.sqrt(v_s.cf[:,1]**2 + v_s.cf[:, 2]**2)
        s_s_sig = s_s.cf[:, 0]

        v_g = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                        bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                        cover=0.95)
        k_flux_g = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True,
                             rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                             cover=0.95)
        s_g = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                        bmodel=True,
                        rangePA=[pa - 10, pa + 10], rangeQ=[q - 0.1, q + 0.1], even=True,
                        cover=0.95)
        flux_g = k_flux_g.cf[:, 0]
        vg_rot = np.sqrt(v_g.cf[:,1]**2 + v_g.cf[:, 2]**2)
        sg_sig = s_g.cf[:, 0]

        return  v_g.velkin, g_velo_err, q, x0, y0, rad, vg_rot, sg_sig, flux_g, n, 3, v_s.velkin, s_velo_err, q, x0, y0, rad, v_s_rot, s_s_sig, flux_s


if __name__ == '__main__':
    mc = False
    if mc == True:
        file = pd.read_csv("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
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

    print("Doing the easy part now...")
    results = MAGPI_kinemetry(source_cat="/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_master_source_catalogue.csv",
                              n_ells=5, n_re=2, SNR_Star=3, SNR_Gas=20)
    file = pd.read_csv("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
    file1 = file[file["MAGPIID"].isin(results[0])]
    file1.to_csv("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_kinemetry_sample_source_catalogue.csv", index=False)
    if mc == True:
        df = pd.DataFrame({"MAGPIID": galaxies,
                           "v_asym_g": GasAsym,
                           "v_asym_g_err": GasAsymErr,
                           "v_asym_s": StarsAsym,
                           "v_asym_s_err": StarsAsymErr,
                           "PA_g": results[1],
                           "PA_s": results[2],
                           "D_PA": results[3],
                           "V_rot_g": results[4],
                           "V_rot_s": results[5],
                           "SNR_g": results[6],
                           "SNR_s": results[7],
                           })
        df.to_csv("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_kinemetry_sample_s05.csv")
        print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
    if mc == False:
        gasasymerr = np.zeros(len(results[0]))
        starasymerr = np.zeros(len(results[0]))
        df = pd.DataFrame({"MAGPIID": results[0],
                           "v_asym_g": results[8],
                           "v_asym_g_err": gasasymerr,
                           "v_asym_s": results[9],
                           "v_asym_s_err": starasymerr,
                           "PA_g": results[1],
                           "PA_s": results[2],
                           "D_PA": results[3],
                           "V_rot_g": results[4],
                           "V_rot_s": results[5],
                           "SNR_g": results[6],
                           "SNR_s": results[7],
                           })
        df.to_csv("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_kinemetry_sample_s05.csv")
        print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
    # BPT_plots("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_kinemetry_sample_BPT_15re.csv",
    #          "/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_csv/MAGPI_kinemetry_sample_s05.csv",nre=1.5)
    print("All done!")
