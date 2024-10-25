import numpy as np
from astropy.io import fits
import astropy.units as u
import os
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from kinemetry import kinemetry
from kinemetry_plots import clean_images_velo
from kinemetry_plots import clean_images_flux
import matplotlib.pyplot as plt
from pafit.fit_kinematic_pa import fit_kinematic_pa

def MAGPI_kinemetry(source_cat, sample=None, n_ells=3, SNR_Star=3, SNR_Gas=20):
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
    zs = []
    re_over_psf=[]
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
            if z[f] > 0.4:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!")
                logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
                continue
            elif z[f] < 0.2:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!")
                logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
                continue
            if quality[f] < 3:
                print(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift failed QC check!")
                logfile.write(f"MAGPIID = {galaxy[f]}, z = {z[f]:.3f}, Redshift failed QC check!\n")
                continue
            elif r50[f]/res_cutoff < cutoff:
                print(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!")
                logfile.write(f"MAGPIID = {galaxy[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!\n")
                continue
            elif galaxy[f] == int("1207128248") or galaxy[f] == int("1506117050") or galaxy[f] == int("1207197197"):
                print(f"MAGPIID = {galaxy[f]}, fixing PA")
                logfile.write(f"MAGPIID = {galaxy[f]}, fixing PA\n")
                pa = pa - 90
            elif galaxy[f] == int("1501180123") or galaxy[f] == int("1502293058") or galaxy[f] == int("1203152196"):
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
        else:
            pass
        star_file = "MAGPI_Absorption_Line/kinematic_maps_spaxels_2MOM_v2.2.1/" + str(
            galaxy[f]) + "_kinematics_ppxf-maps.fits"
        gas_file = "MAGPI_Emission_Line/MAGPI"+field+"/MAGPI" + field + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
            galaxy[f]) + "_GIST_EmissionLines.fits"
        #print(gas_file)
        if os.path.exists(star_file):
            star_file_catch = True
        else:
            print("No stellar kinematics!")
            star_file_catch = False

        if os.path.exists(gas_file):
            gas_file_catch = True
        else:
            print("No gas kinematics!")
            gas_file_catch = False

        # Check to see if there is neither gas or star data
        if star_file_catch==False and gas_file_catch==False:
            print("No kinematics! Skipping "+str(galaxy[f])+"!")
            continue

        # Gas kinemetry
        if star_file_catch==False and gas_file_catch:
            gasfile = fits.open(gas_file)
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data, gasfile[11].data
            gasfile.close()
            g_velo = clean_images_velo(g_velo, pa[f], r50[f], r50[f] * q[f], img_flux=g_flux / g_flux_err,limit=3)
            g_velo_err = clean_images_velo(g_velo_err, pa[f], r50[f], r50[f] * q[f], img_flux=g_flux / g_flux_err,limit=3)
            g_sigma = clean_images_velo(g_sigma, pa[f], r50[f], r50[f] * q[f], img_flux=g_flux / g_flux_err,limit=3)
            g_flux = clean_images_flux(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            clip = np.nanmax(g_flux)
            y0, x0 = g_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Gas SNR = {clip:.2f}...")
            print(f"Max Gas SNR = {clip:.2f}...", file=logfile)
            if clip < SNR_Gas:
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                continue
            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f] + step
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                continue
            print("Doing kinemetry on gas only!")
            print("Doing kinemetry on gas only!", file=logfile)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 2 * r50[f]
            rad = np.arange(start, end, step)

            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, paq=np.array([pa[f]-90, q[f]]), allterms=True)
            vrotg = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2]**2)
            vrotg = vrotg / np.sin(np.arccos(q[f]))

            fig, ax = plt.subplots()
            ax.scatter(kg.rad, vrotg, ec="k", zorder=2, label="Gas")
            ax.plot(kg.rad, vrotg, zorder=1)
            ax.scatter(kg.rad[vrotg == np.nanmax(vrotg)], vrotg[vrotg == np.nanmax(vrotg)], ec="k", label=r"V$_{max}$")
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/rotation_curves/" + str(galaxy[f]) + "_Vrot.pdf",
                        bbox_inches="tight")

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f]+step
            rad = np.arange(start, end, step)
            kgs = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                           bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

            kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

            sigmag = np.nanmean((kg_flux.cf[:,0]*kgs.cf[:, 0])/kg_flux.cf[:,0])
            gs05 = np.sqrt(0.5 * np.nanmax(vrotg) ** 2 + sigmag ** 2)

            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0,360], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                           allterms=True)

            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
            kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
            kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
            kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

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
            pa_g = pa_g
            pa_s = np.nan
            d_pa = np.abs(np.nan)


            vasym_g = kg2+kg3+kg4+kg5
            vasym_g = vasym_g/(4*gs05)
            vasym_g = vasym_g[-1]

            gal_id.append(galaxy[f])
            gas_s05.append(vasym_g)
            star_s05.append(np.nan)
            SNR_g.append(np.nanmean(kg_flux.cf[:,0]))
            SNR_s.append(np.nan)
            v_rot_g.append(np.nanmax(vrotg))
            v_sigma_g.append(sigmag)
            v_rot_s.append(np.nan)
            v_sigma_s.append(np.nan)
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)
            zs.append(z[f])
            re_over_psf.append(r50[f]/res_cutoff)

            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0})
            zeros = np.where(kg.eccano == 0)[0][-2:]
            k1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            x = np.degrees(kg.eccano[zeros[-2]:zeros[-1]])
            ex_mom = kg.ex_mom[zeros[-2]:zeros[-1]]
            vrec = kg.vrec[zeros[-2]:zeros[-1]]
            vv = kg.vv[zeros[-2]:zeros[-1]]
            yEl = kg.Yellip[zeros[-2]:zeros[-1]].astype(int)
            xEl = kg.Xellip[zeros[-2]:zeros[-1]].astype(int)
            chi2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
            chi2 = chi2 + 2 * np.log(len(ex_mom))

            ax1.plot(x, vrec, zorder=1, label='Gas', color="tab:orange")
            ax1.scatter(x, ex_mom, s=7, ec="k", zorder=2, color="tab:orange")
            ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), ec="k", zorder=2, s=2,color='tab:orange')
            ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), zorder=1,color='tab:orange')
            ax2.hlines(0.05,xmin=0,xmax=360,ls="dashed",color="k")
            ax2.hlines(-0.05, xmin=0, xmax=360, ls="dashed", color="k")
            ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=g_velo_err[yEl, xEl] / np.nanmax(k1), color="grey",
                         ls="", zorder=1,capsize=2)
            ax2.hlines(0.05, xmin=0, xmax=360, ls="dashed", color="k")
            ax2.hlines(-0.05, xmin=0, xmax=360, ls="dashed", color="k")
            plt.savefig("MAGPI_Plots/plots/Gas_Stars_Ellipse/" + str(galaxy[f]) + "_gas_star_ellipse.pdf",
                        bbox_inches='tight')

        # Stellar kinemetry
        if star_file_catch and gas_file_catch==False:
            starfile = fits.open(star_file)
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
            starfile.close()

            s_velo = clean_images_velo(s_velo, pa[f], r50[f], r50[f] * q[f], img_flux=s_flux,limit=3)
            s_velo_err = clean_images_velo(s_velo_err, pa[f], r50[f], r50[f] * q[f], img_flux=s_flux,limit=3)
            s_sigma = clean_images_velo(s_sigma, pa[f], r50[f], r50[f] * q[f], img_flux=s_flux,limit=3)

            clip = np.nanmax(s_flux)
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Stellar SNR = {clip:.2f}...")
            print(f"Max Stellar SNR = {clip:.2f}...", file=logfile)
            if clip < SNR_Star:
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its heinous looking")
                continue
            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f]+step
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                continue
            print("Doing kinemetry on stars only!")
            print("Doing kinemetry on stars only!", file=logfile)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 2 * r50[f]
            rad = np.arange(start, end, step)

            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, paq=np.array([pa[f],q[f]]), allterms=True)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f]+step
            rad = np.arange(start, end, step)
            kss = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa[f],q[f]]), even=True)

            ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, paq=np.array([pa[f],q[f]]), even=True)

            sigmas = np.nanmean((ks_flux.cf[:,0]*kss.cf[:, 0])/ks_flux.cf[:,0])
            vrots = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2]**2)
            vrots = vrots / np.sin(np.arccos(q[f]))

            fig, ax = plt.subplots()
            ax.scatter(ks.rad, vrots, ec="k", zorder=2, label="Stars")
            ax.plot(ks.rad, vrots, zorder=1)
            ax.scatter(ks.rad[vrots == np.nanmax(vrots)], vrots[vrots == np.nanmax(vrots)], ec="k", label=r"V$_{max}$",zorder=2)
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/rotation_curves/" + str(galaxy[f]) + "_Vrot.pdf",
                        bbox_inches="tight")

            ss05 = np.sqrt(0.5 * np.nanmax(vrots) ** 2 + sigmas ** 2)
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                           allterms=True)

            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
            ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)

            vasym_s = ks2 + ks3 + ks4 + ks5
            vasym_s = vasym_s / (4 * ss05)
            vasym_s = vasym_s[-1]

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

            pa_g = np.nan
            pa_s = pa_s
            d_pa = np.abs(np.nan)

            gal_id.append(galaxy[f])
            star_s05.append(vasym_s)
            gas_s05.append(np.nan)
            SNR_g.append(np.nan)
            SNR_s.append(np.nanmean(ks_flux.cf[:, 0]))
            v_rot_s.append(np.nanmax(vrots))
            v_sigma_s.append(sigmas)
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)
            v_rot_g.append(np.nan)
            v_sigma_g.append(np.nan)
            zs.append(z[f])
            re_over_psf.append(r50[f] / res_cutoff)

            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0})
            zeros = np.where(ks.eccano == 0)[0][-2:]
            k1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            x = np.degrees(ks.eccano[zeros[-2]:zeros[-1]])
            ex_mom = ks.ex_mom[zeros[-2]:zeros[-1]]
            vrec = ks.vrec[zeros[-2]:zeros[-1]]
            yEl = ks.Yellip[zeros[-2]:zeros[-1]].astype(int)
            xEl = ks.Xellip[zeros[-2]:zeros[-1]].astype(int)
            chi2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
            chi2 = chi2 + 2 * np.log(len(ex_mom))

            ax1.plot(x, vrec, zorder=1, label="Stars", color='tab:blue')
            ax1.scatter(x, ex_mom, s=7, ec="k", zorder=2, color='tab:blue')
            ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), ec="k", zorder=2, s=3,color='tab:blue')
            ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), zorder=1,color="tab:blue")
            ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=s_velo_err[yEl, xEl] / np.nanmax(k1), color="magenta",
                         ls="", zorder=1,capsize=2)
            ax2.hlines(0.05, xmin=0, xmax=360, ls="dashed", color="k")
            ax2.hlines(-0.05, xmin=0, xmax=360, ls="dashed", color="k")
            ax1.legend()
            plt.savefig("MAGPI_Plots/plots/Gas_Stars_Ellipse/" + str(galaxy[f]) + "_gas_star_ellipse.pdf",
                        bbox_inches='tight')
            
        if star_file_catch and gas_file_catch:
            starfile = fits.open(star_file)
            gasfile = fits.open(gas_file)
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
            starfile.close()

            s_velo = clean_images_velo(s_velo, pa[f], r50[f], r50[f] * q[f], img_flux=s_flux,limit=3)
            s_velo_err = clean_images_velo(s_velo_err, pa[f], r50[f], r50[f] * q[f], img_flux=s_flux,limit=3)
            s_sigma = clean_images_velo(s_sigma, pa[f], r50[f], r50[f] * q[f], img_flux=s_flux,limit=3)

            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                                                              gasfile[10].data, gasfile[11].data
            gasfile.close()

            g_velo = clean_images_velo(g_velo, pa[f], r50[f], r50[f] * q[f], img_flux=g_flux / g_flux_err,limit=3)
            g_velo_err = clean_images_velo(g_velo_err, pa[f], r50[f], r50[f] * q[f], img_flux=g_flux / g_flux_err,limit=3)
            g_sigma = clean_images_velo(g_sigma, pa[f], r50[f], r50[f] * q[f], img_flux=g_flux / g_flux_err,limit=3)
            g_flux = clean_images_flux(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            s_clip = np.nanmax(s_flux)
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Stellar SNR = {s_clip:.2f}...")
            if s_clip < SNR_Star or np.isinf(SNR_Star) or np.isnan(SNR_Star):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its stars are heinous looking")
                print("Trying the gas...")
                print("Trying the gas...\n",file=logfile)
                g_clip = np.nanmax(g_flux)
                print(f"Max Gas SNR = {g_clip:.2f}...")
                if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                    print(
                        "Not doing kinemetry on " + str(
                            galaxy[f]) + "because its gas is also heinous looking")
                    continue
                else:
                    print("Doing kinemetry on the gas only!")
                    print("Doing kinemetry on the gas only!", file=logfile)
                    step = (0.65 / 2) / 0.2
                    start = (0.65 / 2) / 0.2 - step
                    end = 1 * r50[f]
                    rad = np.arange(start, end, step)
                    if len(rad) < n_ells:
                        print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                        continue
                    print("Doing kinemetry on gas!")
                    print("Doing kinemetry on gas!", file=logfile)

                    step = (0.65 / 2) / 0.2
                    start = (0.65 / 2) / 0.2 - step
                    end = 2 * r50[f]
                    rad = np.arange(start, end, step)

                    kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, paq=np.array([pa[f]-90, q[f]]), allterms=True)

                    step = (0.65 / 2) / 0.2
                    start = (0.65 / 2) / 0.2 - step
                    end = 1 * r50[f]
                    rad = np.arange(start, end, step)
                    kgs = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                    bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

                    kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                        bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

                    sigmag = np.nanmean((kg_flux.cf[:,0]*kgs.cf[:, 0])/kg_flux.cf[:,0])
                    vrotg = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2]**2)
                    vrotg = vrotg / np.sin(np.arccos(q[f]))

                    fig, ax = plt.subplots()
                    ax.scatter(kg.rad, vrotg, ec="k", zorder=2, label="Stars")
                    ax.plot(kg.rad, vrotg, zorder=1)
                    ax.scatter(kg.rad[vrotg == np.nanmax(vrotg)], vrotg[vrotg == np.nanmax(vrotg)], ec="k",
                               label=r"V$_{max}$",zorder=2)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("MAGPI_Plots/plots/rotation_curves/" + str(galaxy[f]) + "_Vrot.pdf",
                                bbox_inches="tight")

                    gs05 = np.sqrt(0.5 * np.nanmax(vrotg) ** 2 + sigmag ** 2)
                    kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                   allterms=True)

                    kg1 = np.sqrt(kg.cf[:,1]**2 + kg.cf[:, 2] ** 2)
                    kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                    kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
                    kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
                    kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

                    vasym_g = kg2+kg3+kg4+kg5
                    vasym_g = vasym_g / (4 * gs05)
                    vasym_g = vasym_g[-1]

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
                    pa_g = pa_g
                    pa_s = np.nan
                    d_pa = np.abs(np.nan)

                    gal_id.append(galaxy[f])
                    gas_s05.append(vasym_g)
                    star_s05.append(np.nan)
                    SNR_g.append(np.nanmean(kg_flux.cf[:, 0]))
                    SNR_s.append(np.nan)
                    v_rot_g.append(np.nanmax(vrotg))
                    v_rot_s.append(np.nan)
                    v_sigma_g.append(sigmag)
                    v_sigma_s.append(np.nan)
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_pa)
                    zs.append(z[f])
                    re_over_psf.append(r50[f] / res_cutoff)

                    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0})
                    zeros = np.where(kg.eccano == 0)[0][-2:]
                    k1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                    x = np.degrees(kg.eccano[zeros[-2]:zeros[-1]])
                    ex_mom = kg.ex_mom[zeros[-2]:zeros[-1]]
                    vrec = kg.vrec[zeros[-2]:zeros[-1]]
                    vv = kg.vv[zeros[-2]:zeros[-1]]
                    yEl = kg.Yellip[zeros[-2]:zeros[-1]].astype(int)
                    xEl = kg.Xellip[zeros[-2]:zeros[-1]].astype(int)
                    chi2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
                    chi2 = chi2 + 2 * np.log(len(ex_mom))

                    ax1.plot(x, vrec, zorder=1, label='Gas', color="tab:orange")
                    ax1.scatter(x, ex_mom, s=7, ec="k", zorder=2, color="tab:orange")
                    ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), ec="k", zorder=2, s=2,color='tab:orange')
                    ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), zorder=1,color='tab:orange')
                    ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=g_velo_err[yEl, xEl] / np.nanmax(k1),
                                 color="grey",
                                 ls="", zorder=1,capsize=2)
                    ax2.hlines(0.05, xmin=0, xmax=360, ls="dashed", color="k")
                    ax2.hlines(-0.05, xmin=0, xmax=360, ls="dashed", color="k")
                    ax1.legend()
                    plt.savefig("MAGPI_Plots/plots/Gas_Stars_Ellipse/" + str(galaxy[f]) + "_gas_star_ellipse.pdf",
                                bbox_inches='tight')
                    continue

            g_clip = np.nanmax(g_flux)
            print(f"Max Gas SNR = {g_clip:.2f}...")
            if g_clip < SNR_Gas or np.isinf(g_clip) or np.isnan(g_clip):
                print("Not doing kinemetry on " + str(galaxy[f]) + " because its gas is heinous looking")
                print("Trying the stars...")
                print("Trying the stars...", file=logfile)
                s_clip = np.nanmax(s_flux)
                print(f"Max Star SNR = {s_clip:.2f}...")
                if s_clip < SNR_Star or np.isinf(s_clip) or np.isnan(s_clip):
                    print(
                        "Not doing kinemetry on " + str(galaxy[f]) + "because its stars are also heinous looking")
                    continue
                else:
                    step = (0.65 / 2) / 0.2
                    start = (0.65 / 2) / 0.2 - step
                    end = 1 * r50[f]
                    rad = np.arange(start, end, step)
                    if len(rad) < n_ells:
                        print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                        continue
                    print("Doing kinemetry on stars only!")
                    print("Doing kinemetry on stars only!", file=logfile)

                    step = (0.65 / 2) / 0.2
                    start = (0.65 / 2) / 0.2 - step
                    end = 2 * r50[f]
                    rad = np.arange(start, end, step)

                    ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, paq=np.array([pa[f]-90, q[f]]), allterms=True)

                    step = (0.65 / 2) / 0.2
                    start = (0.65 / 2) / 0.2 - step
                    end = 1 * r50[f]
                    rad = np.arange(start, end, step)

                    ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                        bmodel=True, paq=np.array([pa[f]-90, q[f]]),even=True)

                    kss = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                    bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

                    sigmas = np.nanmean((ks_flux.cf[:,0]*kss.cf[:, 0])/ks_flux.cf[:,0])
                    vrots = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2]**2)
                    vrots = vrots / np.sin(np.arccos(q[f]))

                    fig, ax = plt.subplots()
                    ax.scatter(ks.rad, vrots, ec="k", zorder=2, label="Stars")
                    ax.plot(ks.rad, vrots, zorder=1)
                    ax.scatter(ks.rad[vrots==np.nanmax(vrots)],vrots[vrots==np.nanmax(vrots)],ec="k",label=r"V$_{max}$",zorder=2)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("MAGPI_Plots/plots/rotation_curves/" + str(galaxy[f]) + "_Vrot.pdf",
                                bbox_inches="tight")

                    ss05 = np.sqrt(0.5 * np.nanmax(vrots) ** 2 + sigmas ** 2)
                    ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                   bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                                   allterms=True)

                    ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                    ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                    ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
                    ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
                    ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)

                    vasym_s = ks2+ks3+ks4+ks5
                    vasym_s = vasym_s / (4 * ss05)
                    vasym_s = vasym_s[-1]

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
                    pa_g = np.nan
                    pa_s = pa_s
                    d_pa = np.abs(np.nan)

                    gal_id.append(galaxy[f])
                    gas_s05.append(np.nan)
                    star_s05.append(vasym_s)
                    v_rot_s.append(np.nanmax(vrots))
                    v_rot_g.append(np.nan)
                    v_sigma_s.append(sigmas)
                    v_sigma_g.append(np.nan)
                    SNR_g.append(np.nan)
                    SNR_s.append(np.nanmean(ks_flux.cf[:,0]))
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_pa)
                    zs.append(z[f])
                    re_over_psf.append(r50[f] / res_cutoff)

                    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0})
                    zeros = np.where(ks.eccano == 0)[0][-2:]
                    k1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                    x = np.degrees(ks.eccano[zeros[-2]:zeros[-1]])
                    ex_mom = ks.ex_mom[zeros[-2]:zeros[-1]]
                    vrec = ks.vrec[zeros[-2]:zeros[-1]]
                    yEl = ks.Yellip[zeros[-2]:zeros[-1]].astype(int)
                    xEl = ks.Xellip[zeros[-2]:zeros[-1]].astype(int)
                    chi2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
                    chi2 = chi2 + 2 * np.log(len(ex_mom))

                    ax1.plot(x, vrec, zorder=1, label="Stars", color='tab:blue')
                    ax1.scatter(x, ex_mom, s=7, ec="k", zorder=2, color='tab:blue')
                    ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), ec="k", zorder=2, s=3,color="tab:blue")
                    ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), zorder=1,color='tab:blue')
                    ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=s_velo_err[yEl, xEl] / np.nanmax(k1),
                                 color="magenta",
                                 ls="", zorder=1,capsize=2)
                    ax1.legend()
                    ax2.hlines(0.05, xmin=0, xmax=360, ls="dashed", color="k")
                    ax2.hlines(-0.05, xmin=0, xmax=360, ls="dashed", color="k")
                    plt.savefig("MAGPI_Plots/plots/Gas_Stars_Ellipse/" + str(galaxy[f]) + "_gas_star_ellipse.pdf",
                                bbox_inches='tight')
                    continue

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f]+step
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                continue

            print("Doing kinemetry on stars and gas!")
            print("Doing kinemetry on stars and gas!", file=logfile)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 2 * r50[f]
            rad = np.arange(start, end, step)

            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, paq=np.array([pa[f]-90, q[f]]), allterms=True)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f]+step
            rad = np.arange(start, end, step)

            kss = kinemetry(img=s_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

            ks_flux = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

            sigmas = np.nanmean((ks_flux.cf[:,0]*kss.cf[:, 0])/ks_flux.cf[:,0])
            vrots = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2]**2)
            vrots = vrots/np.sin(np.arccos(q[f]))

            fig, ax = plt.subplots()
            ax.scatter(ks.rad, vrots, ec="k", zorder=2, label="Stars")
            ax.plot(ks.rad, vrots, zorder=1)
            ax.scatter(ks.rad[vrots == np.nanmax(vrots)], vrots[vrots == np.nanmax(vrots)], ec="k", label=r"V$_{max}$",
                       zorder=2)
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/rotation_curves/" + str(galaxy[f]) + "_Vrot.pdf",
                        bbox_inches="tight")

            ss05 = np.sqrt(0.5 * np.nanmax(vrots) ** 2 + sigmas ** 2)
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                           allterms=True)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 2 * r50[f]
            rad = np.arange(start, end, step)

            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, paq=np.array([pa[f]-90, q[f]]), allterms=True)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50[f]
            rad = np.arange(start, end, step)

            kgs = kinemetry(img=g_sigma, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                            bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)
            kg_flux = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                bmodel=True, paq=np.array([pa[f]-90, q[f]]), even=True)

            sigmag = np.nanmean((kg_flux.cf[:,0]*kgs.cf[:, 0])/kg_flux.cf[:,0])
            vrotg = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2]**2)
            vrotg = vrotg/np.sin(np.arccos(q[f]))

            fig, ax = plt.subplots()
            ax.scatter(kg.rad, vrotg, ec="k", zorder=2, label="Stars")
            ax.scatter(kg.rad[vrotg == np.nanmax(vrotg)], vrotg[vrotg == np.nanmax(vrotg)], ec="k", label=r"V$_{max}$",zorder=2)
            ax.plot(kg.rad, vrotg, zorder=1)
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/rotation_curves/" + str(galaxy[f]) + "_Vrot.pdf",
                        bbox_inches="tight")

            gs05 = np.sqrt(0.5 * np.nanmax(vrotg) ** 2 + sigmag ** 2)
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q[f] - 0.1, q[f] + 0.1],
                           allterms=True)

            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            ks4 = np.sqrt(ks.cf[:, 6] ** 2 + ks.cf[:, 7] ** 2)
            ks5 = np.sqrt(ks.cf[:, 8] ** 2 + ks.cf[:, 10] ** 2)

            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
            kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
            kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
            kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)

            vasym_s = ks2 + ks3 + ks4 + ks5
            vasym_s = vasym_s / (4 * ss05)
            vasym_s = vasym_s[-1]

            vasym_g = kg2 + kg3 + kg4 + kg5
            vasym_g = vasym_g / (4 * gs05)
            vasym_g = vasym_g[-1]

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

            pa_g = pa_g
            pa_s = pa_s
            d_pa = np.abs(pa_g - pa_s)

            gal_id.append(galaxy[f])
            star_s05.append(vasym_s)
            gas_s05.append(vasym_g)
            v_rot_s.append(np.nanmax(vrots))
            v_rot_g.append(np.nanmax(vrotg))
            v_sigma_g.append(sigmag)
            v_sigma_s.append(sigmas)
            SNR_s.append(np.nanmean(ks_flux.cf[:, 0]))
            SNR_g.append(np.nanmean(kg_flux.cf[:, 0]))
            pa_gs.append(pa_g)
            pa_ss.append(pa_s)
            d_pas.append(d_pa)
            zs.append(z[f])
            re_over_psf.append(r50[f] / res_cutoff)

            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0})
            zeros = np.where(kg.eccano == 0)[0][-2:]
            k1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            x = np.degrees(kg.eccano[zeros[-2]:zeros[-1]])
            ex_mom = kg.ex_mom[zeros[-2]:zeros[-1]]
            vrec = kg.vrec[zeros[-2]:zeros[-1]]
            vv = kg.vv[zeros[-2]:zeros[-1]]
            yEl = kg.Yellip[zeros[-2]:zeros[-1]].astype(int)
            xEl = kg.Xellip[zeros[-2]:zeros[-1]].astype(int)
            chi2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
            chi2 = chi2 + 2 * np.log(len(ex_mom))

            ax1.plot(x, vrec, zorder=1, label='Gas', color="tab:orange")
            ax1.scatter(x, ex_mom, s=7, ec="k", zorder=2,color="tab:orange")
            ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), ec="k", zorder=2, s=2,color="tab:orange")
            ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), zorder=1,color='tab:orange')
            ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=g_velo_err[yEl, xEl] / np.nanmax(k1), color="grey",
                         ls="", zorder=1,capsize=2)

            zeros = np.where(ks.eccano == 0)[0][-2:]
            k1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            x = np.degrees(ks.eccano[zeros[-2]:zeros[-1]])
            ex_mom = ks.ex_mom[zeros[-2]:zeros[-1]]
            vrec = ks.vrec[zeros[-2]:zeros[-1]]
            yEl = ks.Yellip[zeros[-2]:zeros[-1]].astype(int)
            xEl = ks.Xellip[zeros[-2]:zeros[-1]].astype(int)
            chi2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
            chi2 = chi2 + 2 * np.log(len(ex_mom))

            ax1.plot(x, vrec, zorder=1, label="Stars", color='tab:blue')
            ax1.scatter(x, ex_mom, s=7, ec="k", zorder=2,color='tab:blue')
            ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), ec="k", zorder=2, s=3,color="tab:blue")
            ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), zorder=1,color='tab:blue')
            ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=s_velo_err[yEl, xEl] / np.nanmax(k1), color="magenta",
                         ls="", zorder=1,capsize=2)
            ax1.legend()
            ax2.hlines(0.05, xmin=0, xmax=360, ls="dashed", color="k")
            ax2.hlines(-0.05, xmin=0, xmax=360, ls="dashed", color="k")
            plt.savefig("MAGPI_Plots/plots/Gas_Stars_Ellipse/" + str(galaxy[f]) + "_gas_star_ellipse.pdf",
                        bbox_inches='tight')

    results = [gal_id,zs,pa_gs,pa_ss,d_pas,v_rot_g,v_rot_s,v_sigma_g,v_sigma_s,SNR_g,SNR_s,gas_s05,star_s05,re_over_psf]
    for ls in results:
        ls = np.array(ls)

    return results







            
            


            





