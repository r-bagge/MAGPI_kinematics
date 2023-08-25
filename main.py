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


def monte_carlo(args):
    g_model, g_img, g_img_err, k_flux_g, q_g, x0_g, y0_g, rad_g, vg, n, r50, catch = args
    if catch == 1:
        v_asym_gmc_05 = np.zeros(n)
        v_asym_gmc_15 = np.zeros(n)
        v_asym_gmc_fw = np.zeros(n)
        for h in range(n):
            model = g_model
            model[np.isnan(model)] = 0
            model += np.random.normal(loc=0, scale=g_img_err)
            k = kinemetry(img=model, x0=x0_g, y0=y0_g, ntrm=6, plot=False, verbose=False, radius=rad_g, bmodel=True,
                          rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1])
            k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
            k3 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k5 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            v_asym = k5/k1
            #v_asym = (k3+k5) / 2*k1
            try:
                v_asym_gmc_05[h] = v_asym[(rad_g / r50) < 0.5][-1]
            except ValueError:
                v_asym_gmc_05[h] = np.nan
            try:
                v_asym_gmc_15[h] = v_asym[(rad_g / r50) < 1.5][-1]
            except ValueError:
                v_asym_gmc_15[h] = np.nan
            v_asym_gmc_fw[h] = np.nansum(k_flux_g * v_asym) / np.nansum(k_flux_g)
        return v_asym_gmc_05, v_asym_gmc_15, v_asym_gmc_fw
    elif catch == 2:
        v_asym_gmc_05 = np.zeros(n)
        v_asym_gmc_15 = np.zeros(n)
        v_asym_gmc_fw = np.zeros(n)
        for h in range(n):
            model = g_model
            model[np.isnan(model)] = 0
            model += np.random.normal(loc=0, scale=g_img_err)
            k = kinemetry(img=model, x0=x0_g, y0=y0_g, ntrm=11, plot=False, verbose=False, radius=rad_g, bmodel=True,
                          rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1], allterms=True)
            k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
            k2 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k3 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            k4 = np.sqrt(k.cf[:, 6] ** 2 + k.cf[:, 7] ** 2)
            k5 = np.sqrt(k.cf[:, 8] ** 2 + k.cf[:, 10] ** 2)
            v_asym = k5 / k1
            #v_asym = (k2+k3+k4+k5)/(4*k1)
            try:
                v_asym_gmc_05[h] = v_asym[(rad_g / r50) < 0.5][-1]
            except ValueError:
                v_asym_gmc_05[h] = np.nan
            try:
                v_asym_gmc_15[h] = v_asym[(rad_g / r50) < 1.5][-1]
            except ValueError:
                v_asym_gmc_15[h] = np.nan
            v_asym_gmc_fw[h] = np.nansum(k_flux_g * v_asym) / np.nansum(k_flux_g)
        return v_asym_gmc_05, v_asym_gmc_15, v_asym_gmc_fw
    elif catch == 3:
        v_asym_gmc_05 = np.zeros(n)
        v_asym_gmc_15 = np.zeros(n)
        v_asym_gmc_fw = np.zeros(n)
        for h in range(n):
            model = g_model
            model[np.isnan(model)] = 0
            model += np.random.normal(loc=0, scale=g_img_err)
            k = kinemetry(img=model, x0=x0_g, y0=y0_g, ntrm=6, plot=False, verbose=False, radius=rad_g, bmodel=True,
                          rangePA=[0, 360], rangeQ=[q_g - 0.1, q_g + 0.1], fixcen=False)
            k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
            k3 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
            k5 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
            v_asym = k5/k1
            #v_asym = (k3+k5)/(2*k1)
            try:
                v_asym_gmc_05[h] = v_asym[(rad_g / r50) < 0.5][-1]
            except ValueError:
                v_asym_gmc_05[h] = np.nan
            try:
                v_asym_gmc_15[h] = v_asym[(rad_g / r50) < 1.5][-1]
            except ValueError:
                v_asym_gmc_15[h] = np.nan
            v_asym_gmc_fw[h] = np.nansum(k_flux_g * v_asym) / np.nansum(k_flux_g)
        return v_asym_gmc_05, v_asym_gmc_15, v_asym_gmc_fw


def monte_carlo_parallel(pars):
    g_model, g_img, g_img_err, k_flux_g, q_g, x0_g, y0_g, rad_g, vg, n, r50, catch = pars
    cores = None
    if cores is None:
        cores = multiprocessing.cpu_count()
    print(f"Running {n} monte carlos on {cores} Cores!")
    group_size = n // 20
    args = [(g_model, g_img, g_img_err,k_flux_g, q_g, x0_g, y0_g, rad_g, vg, group_size, r50, catch) for _ in range(20)]
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
        mcs = np.empty((3, n), dtype=np.float64)
        for i, r in enumerate(vasyms):
            start = i * group_size
            end = start + group_size
            mcs[:, start:end] = r

    return mcs


def MAGPI_kinemetry_parrallel(args):
    galaxy, pa, q, z, r50, quality, catch = args
    field = str(galaxy)[:4]
    n_re = 2
    res_cutoff = 0.7 / 0.2
    cutoff = 1
    n_ells = 5
    n = 100
    SNR_Gas = 20
    logfile = open("MAGPI_Plots/plots/MAGPI" + field + "/MAGPI" + field + "_logfile.txt", "w")
    if z > 0.35:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!\n")
        return
    elif z < 0.28:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift not in range!\n")
        return
    elif quality < 3:
        print(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift failed QC check!")
        logfile.write(f"MAGPIID = {galaxy}, z = {z:.3f}, Redshift failed QC check!\n")
        return
    elif r50 < cutoff * res_cutoff:
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
    gas_file = "MAGPI_Emission_Lines/MAGPI" + field + "/MAGPI" + field + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
        galaxy) + "_GIST_EmissionLines.fits"

    if os.path.exists(gas_file):
        gas_file_catch = True
    else:
        print("No gas kinematics!")
        logfile.write("No gas kinematics!\n")
        gas_file_catch = False

    # Gas kinemetry using M1
    if gas_file_catch and catch==1:
        gasfile = fits.open(gas_file)
        g_flux, g_flux_err, g_velo, g_velo_err= gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()
        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
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
        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return
        print("Doing kinemetry on gas only!")
        print("Doing kinemetry on gas only!", file=logfile)

        kg_velo = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=6, plot=False, verbose=False, radius=rad,
                            bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1])
        kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                             bmodel=True, paq=np.array([pa,q]), even=True)
        k_flux_g = kg_flux.cf[:,0]
        vg = np.max(np.sqrt(kg_velo.cf[:, 1] ** 2 + kg_velo.cf[:, 2] ** 2))

        return kg_velo.velkin, g_velo, g_velo_err, k_flux_g, q, x0, y0, rad, vg, n, r50, catch

    # Gas kinemetry using M2
    if gas_file_catch and catch==2:
        gasfile = fits.open(gas_file)
        g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()
        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
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
        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return
        print("Doing kinemetry on gas only!")
        print("Doing kinemetry on gas only!", file=logfile)

        kg_velo = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                            bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True)
        kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa, q]), even=True)
        k_flux_g = kg_flux.cf[:, 0]
        vg = np.max(np.sqrt(kg_velo.cf[:, 1] ** 2 + kg_velo.cf[:, 2] ** 2))

        return kg_velo.velkin, g_velo, g_velo_err, k_flux_g, q, x0, y0, rad, vg, n, r50, catch

    # Gas kinemetry using M3
    if gas_file_catch and catch==3:
        gasfile = fits.open(gas_file)
        g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()
        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
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
        start = (0.65 / 2) / 0.2
        step = (0.65 / 2) / 0.2
        end = n_re * r50
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            logfile.write(f"{len(rad)} ellipse/s, Not enough ellipses!\n")
            return
        print("Doing kinemetry on gas only!")
        print("Doing kinemetry on gas only!", file=logfile)

        kg_velo = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=6, plot=False, verbose=False, radius=rad,
                            bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1],fixcen=False)
        kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa, q]), even=True)
        k_flux_g = kg_flux.cf[:, 0]
        vg = np.max(np.sqrt(kg_velo.cf[:, 1] ** 2 + kg_velo.cf[:, 2] ** 2))

        return kg_velo.velkin, g_velo, g_velo_err, k_flux_g, q, x0, y0, rad, vg, n, r50, catch


if __name__ == '__main__':
    just_BPT=False
    catch = 1
    if just_BPT == True:
        print(f"Doing BPT stuff")
        BPT_plots("MAGPI_csv/MAGPI_kinemetry_sample_M2_BPT.csv", "MAGPI_csv/MAGPI_kinemetry_sample_M2.csv", n_re=1.5)
    else:
        if catch==1:
            print("Beginning M1")
            file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
            z = file["z"].to_numpy()
            pa = file["ang_it"].to_numpy()
            q = file["axrat_it"].to_numpy()
            re = file["R50_it"].to_numpy() / 0.2
            quality = file["QOP"].to_numpy()
            galaxy = file["MAGPIID"].to_numpy()
            galaxies = []
            GasAsym_05 = []
            GasAsym_05_Err = []
            GasAsym_15 = []
            GasAsym_15_Err = []
            GasAsym_fw = []
            GasAsym_fw_Err = []
            print("Beginning the hard part...")
            for i in range(len(file)):
                pars = [galaxy[i], pa[i], q[i], z[i], re[i], quality[i], catch]
                args = MAGPI_kinemetry_parrallel(pars)
                if args is None:
                    continue
                mcs = monte_carlo_parallel(args)
                galaxies.append(galaxy[i])
                print(f"Gas Asym 05={np.nanmean(mcs[0]):.2f}")
                GasAsym_05.append(np.nanmean(mcs[0]))
                GasAsym_05_Err.append(np.nanstd(mcs[0]))
                print(f"Gas Asym 15={np.nanmean(mcs[1]):.2f}")
                GasAsym_15.append(np.nanmean(mcs[1]))
                GasAsym_15_Err.append(np.nanstd(mcs[1]))
                print(f"Gas Asym fw={np.nanmean(mcs[2]):.2f}")
                GasAsym_fw.append(np.nanmean(mcs[2]))
                GasAsym_fw_Err.append(np.nanstd(mcs[2]))

            print("Beginning the easy part...")
            results = MAGPI_kinemetry(source_cat="MAGPI_csv/MAGPI_master_source_catalogue.csv", sample=galaxies,
                                      n_ells=5, n_re=2, SNR_Star=3, SNR_Gas=20)
            df = pd.DataFrame({"MAGPIID": galaxies,
                               "v_asym_05": GasAsym_05,
                               "v_asym_05_err": GasAsym_05_Err,
                               "v_asym_15": GasAsym_15,
                               "v_asym_15_err": GasAsym_15_Err,
                               "v_asym_fw": GasAsym_fw,
                               "v_asym_fw_err": GasAsym_fw_Err,
                               "PA_g": results[1],
                               "PA_s": results[2],
                               "D_PA": results[3],
                               "V_rot_g": results[4],
                               "V_rot_s": results[5],
                               "Sigma_g": results[6],
                               "Sigma_s": results[7],
                               "SNR_g": results[8],
                               "SNR_s": results[9],
                               })
            df.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_M1_k51.csv",index=False)
            print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
        catch=2
        if catch==2:
            print("Beginning M2")
            file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
            z = file["z"].to_numpy()
            pa = file["ang_it"].to_numpy()
            q = file["axrat_it"].to_numpy()
            re = file["R50_it"].to_numpy() / 0.2
            quality = file["QOP"].to_numpy()
            galaxy = file["MAGPIID"].to_numpy()
            galaxies = []
            GasAsym_05 = []
            GasAsym_05_Err = []
            GasAsym_15 = []
            GasAsym_15_Err = []
            GasAsym_fw = []
            GasAsym_fw_Err = []
            print("Beginning the hard part...")
            for i in range(len(file)):
                pars = [galaxy[i], pa[i], q[i], z[i], re[i], quality[i], catch]
                args = MAGPI_kinemetry_parrallel(pars)
                if args is None:
                    continue
                mcs = monte_carlo_parallel(args)
                galaxies.append(galaxy[i])
                print(f"Gas Asym 05={np.nanmean(mcs[0]):.2f}")
                GasAsym_05.append(np.nanmean(mcs[0]))
                GasAsym_05_Err.append(np.nanstd(mcs[0]))
                print(f"Gas Asym 15={np.nanmean(mcs[1]):.2f}")
                GasAsym_15.append(np.nanmean(mcs[1]))
                GasAsym_15_Err.append(np.nanstd(mcs[1]))
                print(f"Gas Asym fw={np.nanmean(mcs[2]):.2f}")
                GasAsym_fw.append(np.nanmean(mcs[2]))
                GasAsym_fw_Err.append(np.nanstd(mcs[2]))

            print("Beginning the easy part...")
            results = MAGPI_kinemetry(source_cat="MAGPI_csv/MAGPI_master_source_catalogue.csv", sample=galaxies,
                                      n_ells=5, n_re=2, SNR_Star=3, SNR_Gas=20)
            df = pd.DataFrame({"MAGPIID": galaxies,
                               "v_asym_05": GasAsym_05,
                               "v_asym_05_err": GasAsym_05_Err,
                               "v_asym_15": GasAsym_15,
                               "v_asym_15_err": GasAsym_15_Err,
                               "v_asym_fw": GasAsym_fw,
                               "v_asym_fw_err": GasAsym_fw_Err,
                               "PA_g": results[1],
                               "PA_s": results[2],
                               "D_PA": results[3],
                               "V_rot_g": results[4],
                               "V_rot_s": results[5],
                               "Sigma_g": results[6],
                               "Sigma_s": results[7],
                               "SNR_g": results[8],
                               "SNR_s": results[9],
                               })
            df.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_M2_k51.csv", index=False)
            print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
        catch=3
        if catch==3:
            print("Beginning M3")
            file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
            z = file["z"].to_numpy()
            pa = file["ang_it"].to_numpy()
            q = file["axrat_it"].to_numpy()
            re = file["R50_it"].to_numpy() / 0.2
            quality = file["QOP"].to_numpy()
            galaxy = file["MAGPIID"].to_numpy()
            galaxies = []
            GasAsym_05 = []
            GasAsym_05_Err = []
            GasAsym_15 = []
            GasAsym_15_Err = []
            GasAsym_fw = []
            GasAsym_fw_Err = []
            print("Beginning the hard part...")
            for i in range(len(file)):
                pars = [galaxy[i], pa[i], q[i], z[i], re[i], quality[i], catch]
                args = MAGPI_kinemetry_parrallel(pars)
                if args is None:
                    continue
                mcs = monte_carlo_parallel(args)
                galaxies.append(galaxy[i])
                print(f"Gas Asym 05={np.nanmean(mcs[0]):.2f}")
                GasAsym_05.append(np.nanmean(mcs[0]))
                GasAsym_05_Err.append(np.nanstd(mcs[0]))
                print(f"Gas Asym 15={np.nanmean(mcs[1]):.2f}")
                GasAsym_15.append(np.nanmean(mcs[1]))
                GasAsym_15_Err.append(np.nanstd(mcs[1]))
                print(f"Gas Asym fw={np.nanmean(mcs[2]):.2f}")
                GasAsym_fw.append(np.nanmean(mcs[2]))
                GasAsym_fw_Err.append(np.nanstd(mcs[2]))

            print("Beginning the easy part...")
            results = MAGPI_kinemetry(source_cat="MAGPI_csv/MAGPI_master_source_catalogue.csv", sample=galaxies,
                                      n_ells=5, n_re=2, SNR_Star=3, SNR_Gas=20)
            df = pd.DataFrame({"MAGPIID": galaxies,
                               "v_asym_05": GasAsym_05,
                               "v_asym_05_err": GasAsym_05_Err,
                               "v_asym_15": GasAsym_15,
                               "v_asym_15_err": GasAsym_15_Err,
                               "v_asym_fw": GasAsym_fw,
                               "v_asym_fw_err": GasAsym_fw_Err,
                               "PA_g": results[1],
                               "PA_s": results[2],
                               "D_PA": results[3],
                               "V_rot_g": results[4],
                               "V_rot_s": results[5],
                               "Sigma_g": results[6],
                               "Sigma_s": results[7],
                               "SNR_g": results[8],
                               "SNR_s": results[9],
                               })
            df.to_csv("MAGPI_csv/MAGPI_kinemetry_sample_M3_k51.csv", index=False)
        print(f"Final sample is {len(df):.0f} out of {len(file):.2f}")
        print(f"Doing BPT stuff")
        BPT_plots("MAGPI_csv/MAGPI_kinemetry_sample_M2_BPT.csv", "MAGPI_csv/MAGPI_kinemetry_sample_M2.csv", n_re=2)
