import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from kinemetry import kinemetry
from astropy.cosmology import Planck18 as cosmo
import numpy as np
from astropy.io import fits
import astropy.units as u
import pandas as pd
import os
import cmasher

def clean_images_median(img, pa, a, b, img_err=None,SNR=3,n_re=2):
    y0, x0 = img.shape
    y0, x0 = y0 / 2, x0 / 2
    pa = pa - 90
    pa = np.radians(pa)
    img[0,:] = np.nan
    img[:,0] = np.nan
    for i in range(1,len(img[:, 0])):
        for j in range(1,len(img[0, :])):
            side1 = (((j - x0) * np.cos(pa)) + ((i - y0) * np.sin(pa))) ** 2 / (a ** 2)
            side2 = (((j - x0) * np.sin(pa)) - ((i - y0) * np.cos(pa))) ** 2 / (b ** 2)
            if side1 + side2 > n_re**2:
                img[i, j] = np.nan
            else:
                if img_err is not None and abs(img_err[i, j]) < SNR and i < (len(img[:,0]) - 5) and j < (len(img[0,:])-5) and i > 5 and j > 5:
                    new_img = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],
                               img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
                    new_img = np.nanmedian(new_img)
                    if np.isnan(new_img):
                        img[i,j]=np.nan
                    else:
                        img[i, j] = new_img
    return img

def clean_images(img, pa, a, b, img_err=None,SNR=3,n_re=2):
    y0, x0 = img.shape
    y0, x0 = y0 / 2, x0 / 2
    pa = pa - 90
    pa = np.radians(pa)
    img[0,:] = np.nan
    img[:,0] = np.nan
    for i in range(1,len(img[:, 0])):
        for j in range(1,len(img[0, :])):
            side1 = (((j - x0) * np.cos(pa)) + ((i - y0) * np.sin(pa))) ** 2 / (a ** 2)
            side2 = (((j - x0) * np.sin(pa)) - ((i - y0) * np.cos(pa))) ** 2 / (b ** 2)
            if side1 + side2 > n_re**2:
                img[i, j] = np.nan
            if img_err is not None and abs(img_err[i, j]) < SNR:
                img[i,j]=np.nan
    return img

# Put in try condition for other onedrive
try:
    os.chdir("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW")
except FileNotFoundError:
    os.chdir("/Volumes/DS/MAGPI")
sample = pd.read_csv("MAGPI_csv/MAGPI_kinemetry_sample_M2.csv")
try:
    logfile = open("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Plots/log.txt","w")
except FileNotFoundError:
    logfile = open("MAGPI_Plots/log.txt", "w")
def maps_check():
    sample = pd.read_csv("MAGPI_csv/MAGPI_kinemetry_sample_M2.csv")
    for g in sample["MAGPIID"].to_numpy():
        galaxy = g
        print("Beginning "+str(g)+"...")
        logfile.write(str(g)+"\n")
        field = str(galaxy)[:4]
        master= pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv",skiprows=16)
        master = master[master["MAGPIID"].isin([galaxy])]
        pa = master["ang_it"].to_numpy()[0]
        q = master["axrat_it"].to_numpy()[0]
        r50 = master['R50_it'].to_numpy()[0]/0.2
        z = master["z"].to_numpy()[0]

        try:
            gasfile = fits.open("/Users/z5408076/Documents/OneDrive - UNSW/MAGPI_Maps/MAGPI"+field+"/Emission_Line/MAGPI"+str(galaxy)+"_GIST_EmissionLines.fits")
        except FileNotFoundError:
            try:
                fits.open("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI"+str(galaxy)+"_GIST_EmissionLines.fits")
            except FileNotFoundError:
                print("No gas kinematics!")
            continue
        g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()

        s_velo = clean_images_median(g_velo, pa, r50, r50 * q, img_err=g_flux/g_flux_err)
        y0,x0 = g_flux.shape
        y0,x0 = y0/2,x0/2

        start = r50/2
        step = (0.7/2)/0.2
        end = 1.5 * r50
        rad = np.arange(start, end, step)

        fig,((ax2,ax5),(ax1,ax3),(ax4,ax6)) = plt.subplots(3,2,figsize=(12,14),sharey="row")
        p2=ax2.imshow(s_velo,cmap="cmr.redshift",vmin=-np.nanmax(s_velo),vmax=np.nanmax(s_velo),origin="lower")
        plt.colorbar(p2,ax=ax2,label=r"DATA (MEDIAN) [kms$^{-1}$]",location="top",pad=0.047,fraction=0.05,)
        kg = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[pa-20, pa+20], rangeQ=[q - 0.1, q + 0.1], allterms=True)
        kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
        kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
        kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
        kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
        kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)
        vasym = (kg5+kg4+kg3+kg2)/(4*kg1)
        zeros_kg = np.where(kg.eccano == 0)[0]
        zeros_kg = zeros_kg[1:]
        x = zeros_kg[1]
        y = zeros_kg[0]
        xEl = kg.Xellip[y:x]
        yEl = kg.Yellip[y:x]
        logfile.write("Masking, leaving nans in\n")
        print(rad/r50,file=logfile)
        print(vasym,file=logfile)
        p=ax1.imshow(kg.velkin,cmap="cmr.redshift",vmin=-np.nanmax(kg.velkin),vmax=np.nanmax(kg.velkin),origin="lower")
        ax1.plot(xEl,yEl,c="magenta")
        ax2.plot(xEl, yEl, c="magenta")
        x = zeros_kg[-1]
        y = zeros_kg[-2]
        xEl = kg.Xellip[y:x]
        yEl = kg.Yellip[y:x]
        ax1.plot(xEl, yEl, c="magenta")
        ax2.plot(xEl, yEl, c="magenta")
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # if y0 > x0:
        #     ax1.set_xlim(y0 / 2, 3 * y0 / 2)
        #     ax1.set_ylim(y0 / 2, 3 * y0 / 2)
        # if y0 < x0:
        #     ax1.set_xlim(x0 / 2, 3 * x0 / 2)
        #     ax1.set_ylim(x0 / 2, 3 * x0 / 2)
        plt.colorbar(p, ax=ax1, location="top",pad=0.047,fraction=0.05,label=r"MODEL (MEDIAN) [kms$^{-1}$]")
        ax4.scatter(rad/r50,vasym)
        ax4.set_ylabel(r"v$_{asym}$")
        ax4.set_xlabel(r"R/R$_{50}$")
        ax4.set_xlim(-0.05, 2.05)

        try:
            gasfile = fits.open(
                "/Users/z5408076/Documents/OneDrive - UNSW/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
                    galaxy) + "_GIST_EmissionLines.fits")
        except FileNotFoundError:
            try:
                fits.open(
                    "/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
                        galaxy) + "_GIST_EmissionLines.fits")
            except FileNotFoundError:
                print("No gas kinematics!")
            continue

        g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()

        s_velo = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        y0, x0 = g_flux.shape
        y0, x0 = y0 / 2, x0 / 2

        p5 = ax5.imshow(s_velo, cmap="cmr.redshift", vmin=-np.nanmax(s_velo),vmax=np.nanmax(s_velo), origin="lower")
        plt.colorbar(p5, ax=ax5, label=r"DATA (NO MEDIAN) [kms$^{-1}$]",location="top",pad=0.047,fraction=0.05,)

        kg_2re = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[pa-20, pa+20], rangeQ=[q - 0.1, q + 0.1], allterms=True)
        kg1_2re = np.sqrt(kg_2re.cf[:, 1] ** 2 + kg_2re.cf[:, 2] ** 2)
        kg2_2re = np.sqrt(kg_2re.cf[:, 3] ** 2 + kg_2re.cf[:, 4] ** 2)
        kg3_2re = np.sqrt(kg_2re.cf[:, 5] ** 2 + kg_2re.cf[:, 6] ** 2)
        kg4_2re = np.sqrt(kg_2re.cf[:, 6] ** 2 + kg_2re.cf[:, 7] ** 2)
        kg5_2re = np.sqrt(kg_2re.cf[:, 8] ** 2 + kg_2re.cf[:, 10] ** 2)
        print("No masking",file=logfile)
        vasym_2re = (kg5_2re+kg4_2re+kg3_2re+kg2_2re)/(4*kg1_2re)
        print(rad/r50,file=logfile)
        print(vasym_2re,file=logfile)

        ax6.scatter(rad/r50,vasym_2re)
        ax6.set_xlabel(r"R/R$_{50}$")
        ax6.set_xlim(-0.05, 2.05)
        zeros_kg = np.where(kg.eccano == 0)[0]
        zeros_kg = zeros_kg[1:]
        # 0.5Re
        x = zeros_kg[1]
        y = zeros_kg[0]
        xEl = kg.Xellip[y:x]
        yEl = kg.Yellip[y:x]
        logfile.write("Masking, leaving nans in\n")
        p = ax3.imshow(kg_2re.velkin, cmap="cmr.redshift", vmin=-np.nanmax(kg.velkin),vmax=np.nanmax(kg.velkin), origin="lower")
        ax3.plot(xEl, yEl, c="magenta")
        ax5.plot(xEl, yEl, c="magenta")
        # 1.5Re
        x = zeros_kg[-1]
        y = zeros_kg[-2]
        xEl = kg.Xellip[y:x]
        yEl = kg.Yellip[y:x]
        ax3.plot(xEl, yEl, c="magenta")
        ax5.plot(xEl, yEl, c="magenta")
        # ax3.set_xticks([])
        # ax3.set_yticks([])
        # if y0>x0:
        #     ax3.set_xlim(y0/2,3*y0/2)
        #     ax3.set_ylim(y0/2,3*y0/2)
        # if y0<x0:
        #     ax3.set_xlim(x0/2,3*x0/2)
        #     ax3.set_ylim(x0/2,3*x0/2)
        plt.colorbar(p,ax=ax3,location="top",pad=0.047,fraction=0.05,label=r"MODEL (NO MEDIAN) [kms$^{-1}$]")
        try:
            plt.savefig("/Volumes/LDS/Astro/PhD/MAGPI/plots/Maps_Check/check/"+str(g)+"_check.pdf",bbox_inches='tight')
        except FileNotFoundError:
            plt.savefig("/Volumes/DS/MAGPI/MAGPI_Plots/Maps_Check/check/"+str(g)+"_check.pdf", bbox_inches='tight')

        try:
            gasfile = fits.open(
                "/Users/z5408076/Documents/OneDrive - UNSW/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
                    galaxy) + "_GIST_EmissionLines.fits")
        except FileNotFoundError:
            try:
                fits.open(
                    "/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
                        galaxy) + "_GIST_EmissionLines.fits")
            except FileNotFoundError:
                print("No gas kinematics!")
            continue

        g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()

        s_velo = clean_images_median(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)

        k_M2 = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                         bmodel=True, rangePA=[pa-20, pa+20], rangeQ=[q - 0.1, q + 0.1],
                         allterms=True, ring=0, fixcen=True)
        k_M1 = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=6, plot=False, verbose=False, radius=rad,
                         bmodel=True, rangePA=[pa-20, pa+20], rangeQ=[q - 0.1, q + 0.1],
                         allterms=False, ring=0, fixcen=True)
        k_M3 = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=6, plot=False, verbose=False, radius=rad,
                         bmodel=True, rangePA=[pa-20, pa+20], rangeQ=[q - 0.1, q + 0.1],
                         allterms=False, ring=0, fixcen=False)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        p1=ax1.imshow(s_velo, cmap="cmr.redshift", vmin=-np.nanmax(s_velo),vmax=np.nanmax(s_velo),origin="lower")
        plt.colorbar(p1,ax=ax1,location="top",pad=0.047,fraction=0.05,label=r"DATA")
        zeros_kg = np.where(k_M2.eccano == 0)[0]
        # zeros_kg = zeros_kg[1:]
        x = zeros_kg[1]
        y = zeros_kg[0]
        xEl = k_M2.Xellip[y:x]
        yEl = k_M2.Yellip[y:x]
        # ax2.scatter(xEl,yEl,c="magenta",s=1)
        # ax1.scatter(xEl,yEl,c="magenta",s=1)
        ax2.plot(xEl, yEl, c="magenta")
        ax1.plot(xEl, yEl, c="magenta")
        x = zeros_kg[-1]
        y = zeros_kg[-2]
        xEl = k_M2.Xellip[y:x]
        yEl = k_M2.Yellip[y:x]
        ax2.scatter(xEl[np.isnan(s_velo[yEl.astype(int), xEl.astype(int)])], yEl[np.isnan(s_velo[yEl.astype(int), xEl.astype(int)])], c="cyan", s=2)
        ax1.scatter(xEl[np.isnan(s_velo[yEl.astype(int), xEl.astype(int)])],yEl[np.isnan(s_velo[yEl.astype(int), xEl.astype(int)])], c="cyan",s=2)
        ax1.plot(xEl, yEl, c="magenta")
        ax2.plot(xEl, yEl, c="magenta")
        zeros_kg = np.where(k_M3.eccano == 0)[0]
        r = np.sqrt(((k_M3.xc - x0) ** 2) + ((k_M3.yc - y0) ** 2))
        r = np.insert(r, 0, 0)
        k_M3.rad = np.insert(k_M3.rad, 0, 0)
        k_M3.xc = np.insert(k_M3.xc, 0, x0)
        k_M3.yc = np.insert(k_M3.yc, 0, y0)
        ax3.scatter(k_M3.xc, k_M3.yc, ec="w", c="k", s=5, zorder=3)
        ax3.scatter(k_M3.xc[0], k_M3.yc[0], ec="magenta", c="k", s=8, zorder=3)
        ax3.scatter(k_M3.xc[-1], k_M3.yc[-1], ec="r", c="k", s=8, zorder=3)
        ax3.plot(k_M3.xc, k_M3.yc, c="w", zorder=2)
        p2=ax2.imshow(k_M2.velkin, cmap="cmr.redshift", vmin=-np.nanmax(k_M2.velkin),vmax=np.nanmax(k_M2.velkin),origin="lower")
        plt.colorbar(p2, ax=ax2, location="top", pad=0.047, fraction=0.05, label=r"M2")
        p3=ax3.imshow(k_M3.velkin, cmap="cmr.redshift", vmin=-np.nanmax(k_M3.velkin),vmax=np.nanmax(k_M3.velkin),origin="lower")
        plt.colorbar(p3, ax=ax3, location="top", pad=0.047, fraction=0.05, label=r"M3")
        for ax in [ax1,ax2,ax3]:
            if ax==ax2:
                #ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("x [kpc]")
            ax.set_xlim(x0 / 2, 3 * x0 / 2)
            ax.set_ylim(y0 / 2, 3 * y0 / 2)
            ticks = ax.get_xticks()
            ticks = np.radians(ticks*0.2/3600)*cosmo.luminosity_distance(z).to(u.kpc).value
            x0_ = np.radians(x0 * 0.2 / 3600) * cosmo.luminosity_distance(z).to(u.kpc).value
            #ticks = ticks - x0_
            #ticks = ticks - np.median(ticks)
            print(ticks)
            new_ticks = []
            for i in ticks:
                new_ticks.append(f"{i:.0f}")
            ax.set_xticklabels(new_ticks)
            ticks = ax.get_yticks()
            ticks = np.radians(ticks * 0.2 / 3600) * cosmo.luminosity_distance(z).to(u.kpc).value
            y0_ = np.radians(y0 * 0.2 / 3600) * cosmo.luminosity_distance(z).to(u.kpc).value
            #ticks = ticks - y0_
            print(ticks)
            new_ticks = []
            for i in ticks:
                new_ticks.append(f"{i:.0f}")
            ax.set_yticklabels(new_ticks)
            ax.set_xlabel("x [kpc]")
            ax1.set_ylabel("y [kpc]")
            if ax==ax3:
                ax.yaxis.tick_right()
                ax.set_ylabel("y [kpc]")
                ax.yaxis.set_label_position("right")

        try:
            plt.savefig("/Volumes/LDS/Astro/PhD/MAGPI/plots/Maps_Check/M2_M3/" + str(galaxy) + "_M2_M3.pdf",bbox_inches='tight')
        except FileNotFoundError:
            plt.savefig("/Volumes/DS/MAGPI/MAGPI_Plots/Maps_Check/M2_M3/" + str(galaxy) + "_M2_M3.pdf", bbox_inches='tight')

def vasyms_nans():
    sample = pd.read_csv("MAGPI_csv/MAGPI_kinemetry_sample_M2.csv")
    bpt = pd.read_csv("MAGPI_csv/MAGPI_kinemetry_sample_M2_BPT.csv")
    try:
        logfile = open("/Users/ryanbagge/Library/CloudStorage/OneDrive-UNSW/MAGPI_Plots/log.txt", "w")
    except FileNotFoundError:
        logfile = open("MAGPI_Plots/log.txt", "w")
    n_nans = []
    n_not_nans = []
    vasym_err = []
    for g in sample["MAGPIID"].to_numpy():
        galaxy = g
        field = str(g)[:4]
        master = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
        master = master[master["MAGPIID"].isin([galaxy])]
        pa = master["ang_it"].to_numpy()
        q = master["axrat_it"].to_numpy()[0]
        r50 = master['R50_it'].to_numpy() / 0.2

        vasyms = sample[sample["MAGPIID"].isin([g])]
        bpts = bpt[bpt["MAGPIID"].isin([g])]
        ty = bpt["type(sf+AGN=0, sf=1, sy=2, ln=3)"].to_numpy()
        try:
            gasfile = fits.open(
                "MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
                    galaxy) + "_GIST_EmissionLines.fits")
        except FileNotFoundError:
            gasfile = fits.open(
                "/Users/z5408076/Documents/OneDrive - UNSW/MAGPI_Maps/MAGPI"+field+"/Emission_Line/MAGPI" + str(
                    galaxy) + "_GIST_EmissionLines.fits")
        g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
        gasfile.close()

        g_velo = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = clean_images(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err
        y0, x0 = g_flux.shape
        y0, x0 = y0 / 2, x0 / 2

        start = r50 / 2
        step = (0.7 / 2) / 0.2
        end = 1.5 * r50
        rad = np.arange(start, end, step)

        kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, rangePA=[pa-20, pa+20], rangeQ=[q - 0.1, q + 0.1], allterms=True)
        zeros_kg = np.where(kg.eccano == 0)[0]
        zeros_kg = zeros_kg[1:]
        x = zeros_kg[(rad / r50) < 1.5][-1]
        y = zeros_kg[(rad / r50) < 1.5][-2]
        n_not_nans.append(len(kg.ex_mom[y:x]))
        v_ellip_nans = kg.ex_mom[y:x][np.isnan(kg.ex_mom[y:x])]
        n_nans.append(len(v_ellip_nans))
        vasym_err.append(vasyms["v_asym_15_err"].to_numpy()[0])
    n_not_nans = np.array(n_not_nans)
    n_nans = np.array(n_nans)
    vasym_err = np.array(vasym_err)

    plt.rcParams.update({"font.size":15})
    fig,ax = plt.subplots()
    ax.scatter((n_nans/n_not_nans)[ty==1],vasym_err[ty==1],label="HII")
    ax.scatter((n_nans/n_not_nans)[ty!=1],vasym_err[ty!=1],ec="magenta",color='tab:blue',label="HII+AGN")
    ax.hlines(0.2,xmin=-1,xmax=2,ls="dashed",color="k")
    ax.vlines(0.30,ymin=0,ymax=0.2,ls="dashed",color="k")
    ax.set_ylabel(r"$\sigma (v_{\rm asym})$")
    ax.set_xlabel(r"Frac. of Non-Detect. at 1.5$R_e$")
    ax.set_yscale("log")
    #ax.set_xscale("log")
    ax.set_xlim(-0.1,1.1)
    ax.legend()
    plt.savefig("MAGPI_Plots/Maps_Check/nans_v_sigma.pdf",bbox_inches="tight")

    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots()
    ax.scatter(sample['SNR_g'][ty == 1], vasym_err[ty == 1], label="HII")
    ax.scatter(sample['SNR_g'][ty != 1], vasym_err[ty != 1], ec="magenta", color='tab:blue', label="HII+AGN")
    ax.hlines(0.2, xmin=-5, xmax=90, ls="dashed", color="k")
    #ax.vlines(0.35, ymin=0, ymax=0.2, ls="dashed", color="k")
    ax.set_ylabel(r"$\sigma (v_{\rm asym})$")
    ax.set_xlabel(r"H$\alpha$ SNR")
    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_xlim(-2, 85)
    ax.legend()
    plt.savefig("MAGPI_Plots/Maps_Check/vasym_SNR.pdf", bbox_inches="tight")

    df = pd.DataFrame({"MAGPIID":sample["MAGPIID"].to_numpy(),
                       "v_asym_15_err":vasym_err,
                       "NaNs_at_ellipse":n_nans/n_not_nans})
    df.to_csv("MAGPI_csv/MAGPI_ellipse_nans.csv",index=False)

vasyms_nans()
#maps_check()

