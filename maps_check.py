import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from kinemetry import kinemetry
import numpy as np
from astropy.io import fits
import pandas as pd
import os
import cmasher

def clean_images_median(img, pa, a, b, img_err=None,SNR=3):
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
            if side1 + side2 > 8:
                img[i, j] = np.nan
            else:
                if img_err is not None and abs(img_err[i, j]) < SNR and i < (len(img[:,0]) - 5) and j < (len(img[0,:])-5) and i > 5 and j > 5:
                    new_img = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
                    new_img = np.nanmedian(new_img)
                    if np.isnan(new_img):
                        img[i,j]=np.nan
                    else:
                        img[i, j] = new_img
    return img

def clean_images(img, pa, a, b, img_err=None,SNR=3):
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
            if side1 + side2 > 8:
                img[i, j] = np.nan
            if img_err is not None and abs(img_err[i, j]) < SNR:
                img[i,j]=np.nan
    return img


os.chdir("/Users/z5408076/Documents/OneDrive - UNSW")
sample = pd.read_csv("MAGPI_csv/MAGPI_kinemetry_sample_M2.csv")
logfile = open("/Users/z5408076/Documents/OneDrive - UNSW/MAGPI_Plots/log.txt","w")
n_nans = []
n_not_nans = []
vasym_err = []
for g in sample["MAGPIID"].to_numpy():
    galaxy = g
    print("Beginning "+str(g)+"...")
    logfile.write(str(g)+"\n")
    field = str(galaxy)[:4]
    master= pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv",skiprows=16)
    master = master[master["MAGPIID"].isin([galaxy])]
    pa = master["ang_it"].to_numpy()
    q = master["axrat_it"].to_numpy()[0]
    r50 = master['R50_it'].to_numpy()/0.2

    gasfile = fits.open("MAGPI_Maps/MAGPI"+field+"/Emission_Line/MAGPI"+str(galaxy)+"_GIST_EmissionLines.fits")
    g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
    gasfile.close()

    g_velo = clean_images_median(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_velo_err = clean_images_median(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_flux = clean_images_median(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_flux = g_flux / g_flux_err
    y0,x0 = g_flux.shape
    y0,x0 = y0/2,x0/2

    start = (0.65 / 2) / 0.2
    step = (0.65 / 2) / 0.2
    end = 2 * r50
    rad = np.arange(start, end, step)

    fig,((ax1,ax3),(ax4,ax6)) = plt.subplots(2,2,figsize=(10,6),sharey="row")
    kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True)
    kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
    kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
    kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
    kg4 = np.sqrt(kg.cf[:, 6] ** 2 + kg.cf[:, 7] ** 2)
    kg5 = np.sqrt(kg.cf[:, 8] ** 2 + kg.cf[:, 10] ** 2)
    vasym = (kg5+kg4+kg3+kg2)/(4*kg1)
    logfile.write("Masking, leaving nans in\n")
    print(rad/r50,file=logfile)
    print(vasym,file=logfile)
    p=ax1.imshow(g_velo,cmap="cmr.redshift",vmin=-220,vmax=220,origin="lower")
    ax1.add_patch(Ellipse(xy=(x0, y0), width=1 * r50,height=1 * r50 / q, angle=pa, fc="none", ec="limegreen"))
    ax1.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,height=2 * r50 / q, angle=pa, fc="none", ec="magenta"))
    ax1.add_patch(Ellipse(xy=(x0, y0), width=3 * r50,height=3 * r50 / q, angle=pa, fc="none", ec="limegreen"))
    ax1.add_patch(Ellipse(xy=(x0, y0), width=4 * r50,height=4 * r50 / q, angle=pa, fc="none", ec="magenta"))
    ax1.set_xticks([])
    ax1.set_yticks([])
    if y0>x0:
        ax1.set_xlim(0,2*y0)
        ax1.set_ylim(0,2*y0)
    if y0<x0:
        ax1.set_xlim(0,2*x0)
        ax1.set_ylim(0,2*x0)
    plt.colorbar(p,ax=ax1,location="top",pad=0.047,fraction=0.05,label=r"V [kms$^{-1}$]")
    ax4.scatter(rad/r50,vasym)
    ax4.set_ylabel(r"v$_{asym}$")
    ax4.set_xlabel(r"R/R$_{50}$")
    ax4.set_xlim(-0.05, 2.05)

    gasfile = fits.open(
        "MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
            galaxy) + "_GIST_EmissionLines.fits")
    g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
    gasfile.close()

    g_velo_2re = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_velo_err = clean_images(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_flux = clean_images(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_flux = g_flux / g_flux_err
    y0, x0 = g_flux.shape
    y0, x0 = y0 / 2, x0 / 2

    kg_2re = kinemetry(img=g_velo_2re, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True)
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

    p=ax3.imshow(g_velo_2re,cmap="cmr.redshift",vmin=-220,vmax=220,origin="lower")
    ax3.add_patch(Ellipse(xy=(x0, y0), width=1 * r50,height=1 * r50 / q, angle=pa, fc="none", ec="limegreen"))
    ax3.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,height=2 * r50 / q, angle=pa, fc="none", ec="magenta"))
    ax3.add_patch(Ellipse(xy=(x0, y0), width=3 * r50,height=3 * r50 / q, angle=pa, fc="none", ec="limegreen"))
    ax3.add_patch(Ellipse(xy=(x0, y0), width=4 * r50,height=4 * r50 / q, angle=pa, fc="none", ec="magenta"))
    ax3.set_xticks([])
    ax3.set_yticks([])
    if y0>x0:
        ax3.set_xlim(0,2*y0)
        ax3.set_ylim(0,2*y0)
    if y0<x0:
        ax3.set_xlim(0,2*x0)
        ax3.set_ylim(0,2*x0)
    plt.colorbar(p,ax=ax3,location="top",pad=0.047,fraction=0.05,label=r"V [kms$^{-1}$]")
    plt.savefig("MAGPI_Plots/Maps_Check/"+str(g)+"_check.pdf",bbox_inches='tight')

    vasyms = sample[sample["MAGPIID"].isin([g])]
    gasfile = fits.open(
        "MAGPI_Maps/MAGPI" + field + "/Emission_Line/MAGPI" + str(
            galaxy) + "_GIST_EmissionLines.fits")
    g_flux, g_flux_err, g_velo, g_velo_err = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data
    gasfile.close()

    g_velo = clean_images(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_velo_err = clean_images(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_flux = clean_images(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
    g_flux = g_flux / g_flux_err
    y0, x0 = g_flux.shape
    y0, x0 = y0 / 2, x0 / 2

    start = (0.65 / 2) / 0.2
    step = (0.65 / 2) / 0.2
    end = 2 * r50
    rad = np.arange(start, end, step)

    kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                   bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True)
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

fig,ax = plt.subplots()
ax.scatter(n_nans/n_not_nans,vasym_err)
ax.hlines(0.4,xmin=-1,xmax=2,ls="dashed",color="k")
ax.vlines(0.5,ymin=0,ymax=0.4,ls="dashed",color="k")
ax.set_ylabel(r"$\sigma (v_{\rm asym})$")
ax.set_xlabel("Frac. of NaNs at the ellipse")
ax.set_yscale("log")
#ax.set_xscale("log")
ax.set_xlim(-0.1,1.1)
plt.savefig("MAGPI_Plots/Maps_Check/nans_v_sigma.pdf",bbox_inches="tight")

df = pd.DataFrame({"MAGPIID":sample["MAGPIID"].to_numpy(),
                   "v_asym_15_err":vasym_err,
                   "NaNs_at_ellipse":n_nans/n_not_nans})
df.to_csv("MAGPI_csv/MAGPI_ellipse_nans.csv",index=False)
