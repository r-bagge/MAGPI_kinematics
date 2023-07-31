import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from astropy.io import fits
import astropy.units as u
import os
import shutil
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import glob
from kinemetry import kinemetry
import cmasher as cmr
from dust_extinction.parameter_averages import CCM89
from kinemetry_plots import BPT_plots


def clean_images(img, pa, a, b, img_err=None):
    y0, x0 = img.shape
    y0, x0 = y0 / 2, x0 / 2
    pa = pa - 90
    pa = np.radians(pa)
    for i in range(len(img[:, 0])):
        for j in range(len(img[0, :])):
            side1 = (((j - x0) * np.cos(pa)) + ((i - y0) * np.sin(pa))) ** 2 / (a ** 2)
            side2 = (((j - x0) * np.sin(pa)) - ((i - y0) * np.cos(pa))) ** 2 / (b ** 2)
            if side1 + side2 > 8:
                img[i, j] = np.nan
            if img_err is not None and abs(img_err[i, j]) < 3:
                img[i, j] = np.nan
    return img


def MAGPI_kinemetry(source_cat, n_ells=5, n_re=2, SNR_Star=3, SNR_Gas=20):
    gal_id = []
    v_rot_g = []
    v_rot_s = []
    pa_ss = []
    pa_gs = []
    d_pas = []
    SNR_g = []
    SNR_s = []
    logfile = open("plots/MAGPI_kinemetry.txt", "w")
    master = pd.read_csv(source_cat,skiprows=16)
    z = master["z"].to_numpy()
    r50 = master["R50_it"].to_numpy() / 0.2
    q = master["axrat_it"].to_numpy()
    pa = master["ang_it"].to_numpy()
    quality = master["QOP"].to_numpy()
    galaxy = master["MAGPIID"].to_numpy()
    res_cutoff = 0.70/0.2
    cutoff = 1
    for f in range(len(master)):
        field = str(galaxy[f])[:4]
        if z[f] > 0.35:
            print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!")
            logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
            continue
        elif z[f] < 0.28:
            print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!")
            logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
            continue
        elif quality[f] < 3:
            print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift failed QC check!")
            logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift failed QC check!\n")
            continue
        elif r50[f] < cutoff * res_cutoff:
            print(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!")
            logfile.write(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!\n")
            continue
        elif galaxy[f] == int("1207128248") or galaxy[f] == int("1506117050"):
            print(f"MAGPIID = {galaxy[f]}, fixing PA")
            logfile.write(f"MAGPIID = {galaxy[f]}, fixing PA\n")
            pa[f] = pa[f] - 90
        elif galaxy[f] == int("1207197197"):
            print(f"MAGPIID = {galaxy[f]}, fixing PA")
            logfile.write(f"MAGPIID = {galaxy[f]}, fixing PA\n")
            pa[f] = pa[f] - 180
        elif galaxy[f] == int("1204192193"):
            print(f"MAGPIID = {galaxy[f]}, For Qainhui")
            logfile.write(f"MAGPIID = {galaxy[f]}, For Qainhui\n")
        elif galaxy == int("1501180123") or galaxy == int("1502293058") or galaxy == int("1203152196"):
            print(f"Piece of Shit")
            logfile.write(f"Piece of Shit\n")
            continue
        else:
            print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift passed!")
            print(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.3f}, Res. passed!")
            print(f"MAGPIID = {galaxy[f]} is {(r50[f] / res_cutoff):.3f} beam elements!")
            logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift passed!\n")
            logfile.write(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.3f}, Res. passed!\n")
            logfile.write(f"MAGPIID = {galaxy[f]} is {(r50[f] / res_cutoff):.3f} beam elements!\n")
        star_file = "MAGPI_Absorption_Lines/MAGPI" + field + "/galaxies/" + str(galaxy[f]) + "_kinematics_ppxf-maps.fits"
        gas_file = "MAGPI_Emission_Lines/MAGPI" + field + "/MAGPI" + field + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(galaxy[f]) + "_GIST_EmissionLines.fits"

        if os.path.exists(star_file):
            star_file_catch = True
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
        if star_file_catch==False and gas_file_catch==False:
            print("No kinematics! Skipping "+str(galaxy[f])+"!")
            logfile.write("No kinematics! Skipping "+str(galaxy[f])+"!\n")
            continue

        # Gas kinemetry
        if star_file_catch==False and gas_file_catch:
            gasfile = fits.open(gas_file)
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data, gasfile[11].data
            gasfile.close()
            g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_velo_err = clean_images(g_velo_err, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            clip = np.nanmax(g_flux)
            y0, x0 = g_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Gas SNR = {clip:.2f}...")
            print(f"Max Gas SNR = {clip:.2f}...", file=logfile)
            if clip < SNR_Gas:
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                continue
            print("Doing kinemetry on gas only!")
            print("Doing kinemetry on gas only!", file=logfile)
            g_velo[np.isnan(g_velo)] = 0
            g_velo_err[np.isnan(g_velo_err)] = 0
            g_flux[np.isnan(g_flux)] = 0

            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            v_rot_g.append(np.nanmax(kg1))
            v_rot_s.append(np.nan)
            pa_g = np.nanmedian(kg.pa)
            pa_s = np.nanmedian(np.nan)
            d_pa = np.abs(np.nan)
            gal_id.append(galaxy[f])
            paq = np.array([pa[f],q[f]])
            kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                               paq=paq, allterms=True, even=True)
            kg0 = kg_flux.cf[:,0]
            SNR_g.append(np.nanmean(kg0))
            SNR_s.append(np.nan)
            if pa_g==-120 or pa_g==120:
                pa_gs.append(np.nan)
                d_pas.append(np.nan)
            else:
                pa_gs.append(pa_g)
                pa_ss.append(pa_s)
                d_pas.append(d_pa)

        # Stellar kinemetry
        if star_file_catch and gas_file_catch==False:
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
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                continue
            print("Doing kinemetry on stars only!")
            print("Doing kinemetry on stars only!", file=logfile)
            s_velo[np.isnan(s_velo)] = 0
            s_velo_err[np.isnan(s_velo_err)] = 0
            s_flux[np.isnan(s_flux)] = 0

            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            ks1 = np.sqrt(ks.cf[:,1]**2 + ks.cf[:,2]**2)
            paq = np.array([pa[f], q[f]])
            ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                paq=paq, allterms=True, even=True)
            ks0 = ks_flux.cf[:, 0]
            SNR_g.append(np.nan)
            SNR_s.append(np.nanmean(ks0))
            v_rot_s.append(np.nanmax(ks1))
            v_rot_g.append(np.nan)
            pa_g = np.nanmedian(np.nan)
            pa_s = np.nanmedian(ks.pa)
            d_pa = np.abs(np.nan)
            gal_id.append(galaxy[f])
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)
            
        if star_file_catch and gas_file_catch:
            starfile = fits.open(star_file)
            gasfile = fits.open(gas_file)
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
            starfile.close()
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                                                              gasfile[10].data, gasfile[11].data
            gasfile.close()
            g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_velo_err = clean_images(g_velo_err, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            s_clip = np.nanmax(s_flux)
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Stellar SNR = {s_clip:.2f}...")
            logfile.write(f"Max Stellar SNR = {s_clip:.2f}...\n")
            if s_clip < SNR_Star or np.isinf(SNR_Star) or np.isnan(SNR_Star):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its stars are heinous looking")
                logfile.write(
                    "Not doing kinemetry on " + str(galaxy[f]) + " because its stars are heinous looking\n")
                print("Trying the gas...")
                print("Trying the gas...\n",file=logfile)
                g_clip = np.nanmax(g_flux)
                print(f"Max Gas SNR = {g_clip:.2f}...")
                logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
                if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                    print(
                        "Not doing kinemetry on " + str(
                            galaxy[f]) + "because its gas is also heinous looking")
                    logfile.write(
                        "Not doing kinemetry on " + str(
                            galaxy[f]) + " because its gas is also heinous looking\n")
                    continue
                else:
                    print("Doing kinemetry on the gas only!")
                    print("Doing kinemetry on the gas only!", file=logfile)
                    start = (0.65 / 2) / 0.2
                    step = (0.65 / 2) / 0.2
                    end = n_re * r50[f]
                    rad = np.arange(start, end, step)
                    if len(rad) < n_ells:
                        print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                        logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                        continue
                    print("Doing kinemetry on gas!")
                    print("Doing kinemetry on gas!", file=logfile)
                    g_velo[np.isnan(g_velo)] = 0
                    g_velo_err[np.isnan(g_velo_err)] = 0
                    g_flux[np.isnan(g_flux)] = 0

                    kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
                    kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nan)

                    paq = np.array([pa[f], q[f]])
                    kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                        paq=paq, allterms=True, even=True)
                    kg0 = kg_flux.cf[:, 0]
                    SNR_g.append(np.nanmean(kg0))
                    SNR_s.append(np.nan)
                    pa_g = np.nanmedian(kg.pa)
                    pa_s = np.nanmedian(np.nan)
                    d_pa = np.abs(np.nan)
                    gal_id.append(galaxy[f])
                    if pa_g == -120 or pa_g == 120:
                        pa_gs.append(np.nan)
                        d_pas.append(np.nan)
                    else:
                        pa_gs.append(pa_g)
                        pa_ss.append(pa_s)
                        d_pas.append(d_pa)

                    continue

            g_clip = np.nanmax(g_flux)
            print(f"Max Gas SNR = {g_clip:.2f}...")
            logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
            if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                print("Not Plotting or doing Kinemetry on " + str(galaxy[f]) + " because its gas is heinous looking")
                logfile.write(
                    "Not Plotting or doing Kinemetry on " + str(galaxy[f]) + " because its gas is heinous looking\n")
                print("Trying the stars...")
                print("Trying the stars...", file=logfile)
                s_clip = np.nanmax(s_flux)
                print(f"Max Star SNR = {s_clip:.2f}...")
                logfile.write(f"Max Star SNR = {s_clip:.2f}...\n")
                if s_clip < SNR_Star or np.isinf(s_clip) or np.isnan(s_clip):
                    print(
                        "Not doing kinemetry on " + str(galaxy[f]) + "because its gas are also heinous looking")
                    logfile.write(
                        "Not doing kinemetry on " + str(
                            galaxy[f]) + " because its stars are also heinous looking\n")
                    continue
                else:
                    start = (0.65 / 2) / 0.2
                    step = (0.65 / 2) / 0.2
                    end = n_re * r50[f]
                    rad = np.arange(start, end, step)
                    if len(rad) < n_ells:
                        print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                        logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                        continue
                    print("Doing kinemetry on stars only!")
                    print("Doing kinemetry on stars only!", file=logfile)
                    s_velo[np.isnan(s_velo)] = 0
                    s_velo_err[np.isnan(s_velo_err)] = 0
                    s_flux[np.isnan(s_flux)] = 0

                    ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
                    ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                    v_rot_s.append(np.nanmax(ks1))
                    v_rot_g.append(np.nan)

                    paq = np.array([pa[f], q[f]])
                    ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                        paq=paq, allterms=True, even=True)
                    ks0 = ks_flux.cf[:, 0]
                    SNR_g.append(np.nan)
                    SNR_s.append(np.nanmean(ks0))

                    pa_g = np.nanmedian(np.nan)
                    pa_s = np.nanmedian(ks.pa)
                    d_pa = np.abs(np.nan)
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_pa)

                    gal_id.append(galaxy[f])
                    continue

            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                continue
            s_velo[np.isnan(s_velo)] = 0
            g_velo[np.isnan(g_velo)] = 0
            g_velo_err[np.isnan(g_velo_err)] = 0
            g_flux[np.isnan(g_flux)] = 0
            s_flux[np.isnan(s_flux)] = 0

            print("Doing kinemetry on stars and gas!")
            print("Doing kinemetry on stars and gas!", file=logfile)
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            ks1 = np.sqrt(ks.cf[:,1]**2+ks.cf[:,2]**2)
            v_rot_s.append(np.nanmax(ks1))
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            v_rot_g.append(np.nanmax(kg1))

            paq = np.array([pa[f], q[f]])
            ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                paq=paq, allterms=True, even=True)
            ks0 = ks_flux.cf[:, 0]
            SNR_s.append(np.nanmean(ks0))
            kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                paq=paq, allterms=True, even=True)
            kg0 = kg_flux.cf[:, 0]
            SNR_g.append(np.nanmean(kg0))

            pa_g = np.nanmedian(kg.pa)
            pa_s = np.nanmedian(ks.pa)
            d_pa = np.abs(pa_g - pa_s)
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)

            gal_id.append(galaxy[f])

    results = [gal_id,pa_gs,pa_ss,d_pas,v_rot_g,v_rot_s,SNR_g,SNR_s]
    for ls in results:
        ls = np.array(ls)

    return results


def radial_rotation(file):
    sample = pd.read_csv(file)
    sample = sample[sample["v_asym_s"] / sample["v_asym_s_err"] > 3]
    source_cat = pd.read_csv("MAGPI_csv/MAGPI_kinemetry_sample_source_catalogue.csv")
    source_cat = source_cat[source_cat["MAGPIID"].isin(sample["MAGPIID"])]
    galaxy = sample['MAGPIID'].to_numpy()
    pa = source_cat["ang_it"].to_numpy()
    q = source_cat["axrat_it"].to_numpy()
    r50 = source_cat["R50_it"].to_numpy() / 0.2
    z = source_cat["z"].to_numpy()
    pix = np.radians((0.33 / 0.2)) * cosmo.luminosity_distance(z).to(u.kpc).value
    vg = sample['v_asym_g'].to_numpy()
    vs = sample["v_asym_s"].to_numpy()
    srad = np.zeros(len(sample))
    srot = np.zeros(len(sample))
    grad = np.zeros(len(sample))
    grot = np.zeros(len(sample))
    for i in range(len(sample["MAGPIID"])):
        if np.isnan(vg[i]):
            print("Do kinemetry only on stars")
            field = str(galaxy[i])[:4]
            starfile = fits.open(
                "MAGPI_Absorption_Lines/MAGPI" + field + "/galaxies/" + str(galaxy[i]) + "_kinematics_ppxf-maps.fits")
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
            starfile.close()
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = 2 * r50[i]
            rad = np.arange(start, end, step)
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True,
                           cover=0.95)
            if np.abs(np.nanmin(ks.cf[:, 1])) > np.abs(np.nanmax(ks.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmin(ks.cf[:,1]):.2f}")
                srad[i] = np.nanmin(ks.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(ks.cf[:,2]):.2f}")
                srot[i] = np.nanmax(ks.cf[:, 2])
            if np.abs(np.nanmin(ks.cf[:, 1])) < np.abs(np.nanmax(ks.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmax(ks.cf[:,1]):.2f}")
                srad[i] = np.nanmax(ks.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(ks.cf[:,2]):.2f}")
                srot[i] = np.nanmax(ks.cf[:, 2])
            else:
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmax(ks.cf[:,1]):.2f}")
                srad[i] = np.nanmax(ks.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(ks.cf[:,2]):.2f}")
                srot[i] = np.nanmax(ks.cf[:, 1])
        if np.isnan(vs[i]):
            print("Do kinemetry only on gas")
            field = str(galaxy[i])[:4]
            gasfile = fits.open(
                "MAGPI_Emission_Lines/MAGPI" + field + "/MAGPI" + field + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
                    galaxy[i]) + "_GIST_EmissionLines.fits")
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                gasfile[10].data, gasfile[11].data
            gasfile.close()
            g_velo = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_velo_err = clean_images(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_flux = clean_images(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = 2 * r50[i]
            rad = np.arange(start, end, step)
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True,
                           cover=0.95)
            if np.abs(np.nanmin(kg.cf[:, 1])) > np.abs(np.nanmax(kg.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmin(kg.cf[:,1]):.2f}")
                grad[i] = np.nanmin(kg.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(kg.cf[:,2]):.2f}")
                grot[i] = np.nanmax(kg.cf[:, 2])
            if np.abs(np.nanmin(kg.cf[:, 1])) < np.abs(np.nanmax(kg.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmax(kg.cf[:,1]):.2f}")
                grad[i] = np.nanmax(kg.cf[:, 1])
                # print("Stellar Rot Velocity")
                grot[i] = np.nanmax(kg.cf[:, 2])
                # print(f"{np.nanmax(kg.cf[:,2]):.2f}")
        if not np.isnan(vs[i]) and not np.isnan(vg[i]):
            print("Do kinemetry on both")
            field = str(galaxy[i])[:4]
            starfile = fits.open(
                "MAGPI_Absorption_Lines/MAGPI" + field + "/galaxies/" + str(galaxy[i]) + "_kinematics_ppxf-maps.fits")
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
            starfile.close()
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = 2 * r50[i]
            rad = np.arange(start, end, step)
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True,
                           cover=0.95)
            if np.abs(np.nanmin(ks.cf[:, 1])) > np.abs(np.nanmax(ks.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmin(ks.cf[:,1]):.2f}")
                srad[i] = np.nanmin(ks.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(ks.cf[:,2]):.2f}")
                srot[i] = np.nanmax(ks.cf[:, 2])
            if np.abs(np.nanmin(ks.cf[:, 1])) < np.abs(np.nanmax(ks.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmax(ks.cf[:,1]):.2f}")
                srad[i] = np.nanmax(ks.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(ks.cf[:,2]):.2f}")
                srot[i] = np.nanmax(ks.cf[:, 2])
            field = str(galaxy[i])[:4]
            gasfile = fits.open(
                "MAGPI_Emission_Lines/MAGPI" + field + "/MAGPI" + field + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
                    galaxy[i]) + "_GIST_EmissionLines.fits")
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                gasfile[10].data, gasfile[11].data
            gasfile.close()
            g_velo = clean_images(g_velo, pa[i], r50[i], r50[i] * q[i], img_err=g_flux / g_flux_err)
            g_velo_err = clean_images(g_velo_err, pa[i], r50[i], r50[i] * q[i], img_err=g_flux / g_flux_err)
            g_flux = clean_images(g_flux, pa[i], r50[i], r50[i] * q[i], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = 2 * r50[i]
            rad = np.arange(start, end, step)
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True,
                           cover=0.95)
            if np.abs(np.nanmin(kg.cf[:, 1])) > np.abs(np.nanmax(kg.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmin(kg.cf[:,1]):.2f}")
                grad[i] = np.nanmin(kg.cf[:, 1])
                # print("Stellar Rot Velocity")
                # print(f"{np.nanmax(kg.cf[:,2]):.2f}")
                grot[i] = np.nanmax(kg.cf[:, 2])
            if np.abs(np.nanmin(kg.cf[:, 1])) < np.abs(np.nanmax(kg.cf[:, 1])):
                # print("Stellar Radial Velocity")
                # print(f"{np.nanmax(kg.cf[:,1]):.2f}")
                grad[i] = np.nanmax(kg.cf[:, 1])
                # print("Stellar Rot Velocity")
                grot[i] = np.nanmax(kg.cf[:, 2])
                # print(f"{np.nanmax(kg.cf[:,2]):.2f}")
    results = [srad,srot,grad,grot]
    return results







            
            


            





