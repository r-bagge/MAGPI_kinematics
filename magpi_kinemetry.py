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


def MAGPI_kinemetry(source_cat, sample=None, n_ells=5, n_re=2, SNR_Star=3, SNR_Gas=20):
    gal_id = []
    v_rot_g = []
    v_rot_s = []
    v_sigma_g = []
    v_sigma_s = []
    pa_ss = []
    pa_gs = []
    d_pas = []
    gas_s05 = []
    star_s05 = []
    SNR_g = []
    SNR_s = []
    logfile = open("MAGPI_csv/MAGPI_kinemetry.txt", "w")
    master = pd.read_csv(source_cat,skiprows=16)
    if sample is not None:
        master = master[master["MAGPIID"].isin(sample)]
    else:
        pass
    z = master["z"].to_numpy()
    r50 = master["R50_it"].to_numpy() / 0.2
    q = master["axrat_it"].to_numpy()
    pa = master["ang_it"].to_numpy()
    quality = master["QOP"].to_numpy()
    galaxy = master["MAGPIID"].to_numpy()
    DL = cosmo.luminosity_distance(z).to(u.kpc).value
    res_cutoff = 0.7/0.2
    cutoff = 1
    for f in range(len(master)):
        field = str(galaxy[f])[:4]
        if sample is None:
            if z[f] > 0.35:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!")
                #logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
                continue
            elif z[f] < 0.28:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!")
                #logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
                continue
            elif quality[f] < 3:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift failed QC check!")
                #logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift failed QC check!\n")
                continue
            elif r50[f] < cutoff * res_cutoff:
                print(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!")
                #logfile.write(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!\n")
                continue
            elif galaxy[f] == int("1207128248") or galaxy[f] == int("1506117050"):
                print(f"MAGPIID = {galaxy[f]}, fixing PA")
                #logfile.write(f"MAGPIID = {galaxy[f]}, fixing PA\n")
                pa[f] = pa[f] - 90
            elif galaxy[f] == int("1207197197"):
                print(f"MAGPIID = {galaxy[f]}, fixing PA")
                #logfile.write(f"MAGPIID = {galaxy[f]}, fixing PA\n")
                pa[f] = pa[f] - 90
            elif galaxy[f] == int("1501180123") or galaxy[f] == int("1502293058"):
                print(f"Piece of Shit")
                #logfile.write(f"MAGPIID = {galaxy[f]}, For Qainhui\n")
                continue
            else:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift passed!")
                print(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.3f}, Res. passed!")
                print(f"MAGPIID = {galaxy[f]} is {(r50[f] / res_cutoff):.3f} beam elements!")
                #logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift passed!\n")
                #logfile.write(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.3f}, Res. passed!\n")
                #logfile.write(f"MAGPIID = {galaxy[f]} is {(r50[f] / res_cutoff):.3f} beam elements!\n")
        else:
            pass
        star_file = "/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Maps/MAGPI" + field + "/Absorption_Line/" + str(galaxy[f]) + "_kinematics_ppxf-maps.fits"
        gas_file = "/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(galaxy[f]) + "_GIST_EmissionLines.fits"
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
        if star_file_catch==False and gas_file_catch==False:
            print("No kinematics! Skipping "+str(galaxy[f])+"!")
            #logfile.write("No kinematics! Skipping "+str(galaxy[f])+"!\n")
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
                #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                continue
            print("Doing kinemetry on gas only!")
            print("Doing kinemetry on gas only!", file=logfile)
            g_velo[np.isnan(g_velo)] = 0
            g_sigma[np.isnan(g_sigma)] = 0
            g_velo_err[np.isnan(g_velo_err)] = 0
            g_flux[np.isnan(g_flux)] = 0

            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            kgs = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)
            kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[pa[f]-10, pa[f]+10], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)

            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            #kg1 = kg.cf[:,2]
            kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
            kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
            kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
            kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

            kgs0 = np.nanmean(kgs.cf[:,0][(rad/r50[f]) < 1])
            gs05 = np.sqrt(0.5*np.nanmax(kg1)**2 + kgs0**2)
            vasym_g = kg2+kg3+kg4+kg5
            vasym_g = vasym_g/(4*gs05)
            vasym_g[np.isnan(vasym_g)]=0
            try:
                vasym_g = vasym_g[(rad / r50[f]) < 1][-1]
            except IndexError:
                gas_s05.append(np.nan)
            gas_s05.append(vasym_g)
            star_s05.append(np.nan)
            SNR_g.append(np.nanmean(kg_flux.cf[:,0]))
            SNR_s.append(np.nan)
            v_rot_g.append(np.nanmax(kg1))
            v_sigma_g.append(kgs0)
            v_rot_s.append(np.nan)
            v_sigma_s.append(np.nan)
            pa_g = np.nanmedian(kg.pa)
            pa_s = np.nanmedian(np.nan)
            d_pa = np.abs(np.nan)
            gal_id.append(galaxy[f])
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
                #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking\n")
                continue
            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                continue
            print("Doing kinemetry on stars only!")
            print("Doing kinemetry on stars only!", file=logfile)
            s_velo[np.isnan(s_velo)] = 0
            s_velo_err[np.isnan(s_velo_err)] = 0
            s_sigma[np.isnan(s_sigma)] = 0
            s_flux[np.isnan(s_flux)] = 0

            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            kss = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)
            ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                even=True)
            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            #ks1 = ks.cf[:,1]
            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
            ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)

            vasym_s = ks2 + ks3 + ks4 + ks5
            kss0 = np.nanmean(kss.cf[:, 0][(rad / r50[f]) < 1])
            ss05 = np.sqrt(0.5 * np.nanmax(ks1) ** 2 + kss0 ** 2)
            vasym_s = vasym_s / (4 * ss05)
            vasym_s[np.isnan(vasym_s)] = 0
            try:
                vasym_s = vasym_s[(rad / r50[f]) < 1][-1]
            except IndexError:
                star_s05.append(np.nan)
            star_s05.append(vasym_s)
            gas_s05.append(np.nan)
            SNR_g.append(np.nan)
            SNR_s.append(np.nanmean(ks_flux.cf[:,0]))

            v_rot_s.append(np.nanmax(ks1))
            v_sigma_s.append(kss0)
            pa_g = np.nanmedian(np.nan)
            pa_s = np.nanmedian(ks.pa)
            d_pa = np.abs(np.nan)

            gal_id.append(galaxy[f])
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)
            v_rot_g.append(np.nan)
            v_sigma_s.append(np.nan)
            
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
            #logfile.write(f"Max Stellar SNR = {s_clip:.2f}...\n")
            if s_clip < SNR_Star or np.isinf(SNR_Star) or np.isnan(SNR_Star):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its stars are heinous looking")
                #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its stars are heinous looking\n")
                print("Trying the gas...")
                print("Trying the gas...\n",file=logfile)
                g_clip = np.nanmax(g_flux)
                print(f"Max Gas SNR = {g_clip:.2f}...")
                #logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
                if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                    print(
                        "Not doing kinemetry on " + str(
                            galaxy[f]) + "because its gas is also heinous looking")
                    #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its gas is also heinous looking\n")
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
                        #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                        continue
                    print("Doing kinemetry on gas!")
                    print("Doing kinemetry on gas!", file=logfile)
                    g_velo[np.isnan(g_velo)] = 0
                    g_sigma[np.isnan(g_sigma)] = 0
                    g_velo_err[np.isnan(g_velo_err)] = 0
                    g_flux[np.isnan(g_flux)] = 0

                    kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
                    kgs = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                    bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)
                    kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                        bmodel=True, rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                        even=True)

                    kg1 = np.sqrt(kg.cf[:,1]**2 + kg.cf[:, 2] ** 2)
                    #kg1 = kg.cf[:,2]
                    kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                    kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
                    kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
                    kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

                    kgs0 = np.nanmean(kgs.cf[:, 0][(rad / r50[f]) < 1])
                    gs05 = np.sqrt(0.5 * np.nanmax(kg1) ** 2 + kgs0 ** 2)
                    vasym_g = kg2+kg3+kg4+kg5
                    vasym_g = vasym_g / (4 * gs05)
                    vasym_g[np.isnan(vasym_g)] = 0
                    try:
                        vasym_g = vasym_g[(rad / r50[f]) < 1][-1]
                    except IndexError:
                        gas_s05.append(np.nan)
                    gas_s05.append(vasym_g)
                    star_s05.append(np.nan)
                    SNR_g.append(np.nanmean(kg_flux.cf[:,0]))
                    SNR_s.append(np.nan)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nan)
                    v_sigma_g.append(kgs0)
                    v_sigma_s.append(np.nan)
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
            #logfile.write(f"Max Gas SNR = {g_clip:.2f}...\n")
            if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its gas is heinous looking")
                #logfile.write("Not Plotting or doing Kinemetry on " + str(galaxy[f]) + " because its gas is heinous looking\n")
                print("Trying the stars...")
                print("Trying the stars...", file=logfile)
                s_clip = np.nanmax(s_flux)
                print(f"Max Star SNR = {s_clip:.2f}...")
                #logfile.write(f"Max Star SNR = {s_clip:.2f}...\n")
                if s_clip < SNR_Star or np.isinf(s_clip) or np.isnan(s_clip):
                    print(
                        "Not doing kinemetry on " + str(galaxy[f]) + "because its stars are also heinous looking")
                    #logfile.write("Not doing kinemetry on " + str(galaxy[f]) + " because its stars are also heinous looking\n")
                    continue
                else:
                    start = (0.65 / 2) / 0.2
                    step = (0.65 / 2) / 0.2
                    end = n_re * r50[f]
                    rad = np.arange(start, end, step)
                    if len(rad) < n_ells:
                        print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                        #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                        continue
                    print("Doing kinemetry on stars only!")
                    print("Doing kinemetry on stars only!", file=logfile)
                    s_velo[np.isnan(s_velo)] = 0
                    s_velo_err[np.isnan(s_velo_err)] = 0
                    s_sigma[np.isnan(s_sigma)] = 0
                    s_flux[np.isnan(s_flux)] = 0

                    ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
                    kss = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                    bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)
                    ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                        bmodel=True, rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                        even=True)
                    ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                    #ks1 = ks.cf[:,2]
                    ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                    ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
                    ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
                    ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)

                    kss0 = np.nanmean(kss.cf[:, 0][(rad / r50[f]) < 1])
                    ss05 = np.sqrt(0.5 * np.nanmax(ks1) ** 2 + kss0 ** 2)
                    vasym_s = ks2+ks3+ks4+ks5
                    vasym_s = vasym_s / (4 * ss05)
                    vasym_s[np.isnan(vasym_s)] = 0
                    try:
                        vasym_s = vasym_s[(rad / r50[f]) < 1][-1]
                    except IndexError:
                        star_s05.append(np.nan)
                    star_s05.append(vasym_s)
                    gas_s05.append(np.nan)
                    SNR_g.append(np.nan)
                    SNR_s.append(np.nanmean(ks_flux.cf[:,0]))

                    v_rot_s.append(np.nanmax(ks1))
                    v_rot_g.append(np.nan)
                    v_sigma_s.append(kss0)
                    v_sigma_g.append(np.nan)
                    pa_g = np.nanmedian(np.nan)
                    pa_s = np.nanmedian(ks.pa)
                    d_pa = np.abs(np.nan)
                    gal_id.append(galaxy[f])
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_pa)
                    continue

            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                #logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
                continue
            s_velo[np.isnan(s_velo)] = 0
            g_velo[np.isnan(g_velo)] = 0
            s_sigma[np.isnan(s_sigma)] = 0
            g_sigma[np.isnan(g_sigma)] = 0
            g_velo_err[np.isnan(g_velo_err)] = 0
            g_flux[np.isnan(g_flux)] = 0
            s_flux[np.isnan(s_flux)] = 0

            print("Doing kinemetry on stars and gas!")
            print("Doing kinemetry on stars and gas!", file=logfile)
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            kss = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)
            ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                even=True)

            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True)
            kgs = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1], even=True)
            kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                even=True)

            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            #ks1 = ks.cf[:,2]
            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
            ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)

            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            #kg1 = kg.cf[:,2]
            kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
            kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
            kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
            kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

            kss0 = np.nanmean(kss.cf[:, 0][(rad / r50[f]) < 1])
            ss05 = np.sqrt(0.5 * np.nanmax(ks1) ** 2 + kss0 ** 2)

            vasym_s = ks2 + ks3 + ks4 + ks5
            vasym_s = vasym_s / (4 * ss05)
            vasym_s[np.isnan(vasym_s)] = 0
            try:
                vasym_s = vasym_s[(rad / r50[f]) < 1][-1]
            except IndexError:
                star_s05.append(np.nan)
            star_s05.append(vasym_s)
            v_rot_s.append(np.nanmax(ks1))
            v_sigma_s.append(kss0)
            SNR_s.append(np.nanmean(ks_flux.cf[:, 0]))

            kgs0 = np.nanmean(kgs.cf[:, 0][(rad / r50[f]) < 1])
            gs05 = np.sqrt(0.5 * np.nanmax(kg1) ** 2 + kgs0 ** 2)
            vasym_g = kg2 + kg3 + kg4 + kg5
            vasym_g = vasym_g / (4 * gs05)
            vasym_g[np.isnan(vasym_g)] = 0
            try:
                vasym_g = vasym_g[(rad / r50[f]) < 1][-1]
            except IndexError:
                gas_s05.append(np.nan)
            gas_s05.append(vasym_g)
            SNR_g.append(np.nanmean(kg_flux.cf[:,0]))

            v_rot_g.append(np.nanmax(kg1))
            v_sigma_g.append(kgs0)
            pa_g = np.nanmedian(kg.pa)
            pa_s = np.nanmedian(ks.pa)
            d_pa = np.abs(pa_g - pa_s)

            gal_id.append(galaxy[f])
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)

    results = [gal_id,pa_gs,pa_ss,d_pas,v_rot_g,v_rot_s,v_sigma_g,v_sigma_s,SNR_g,SNR_s,gas_s05,star_s05]
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
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True)
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
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True)
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
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True)
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
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[i] - 0.1, q[i] + 0.1], allterms=True)
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







            
            


            





