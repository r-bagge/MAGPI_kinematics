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

def dust_corr(flux, balmer, lam, model):
    ext = model(Rv=3.1)
    ebv = (2.5 / 1.163) * np.log10(balmer / 2.86)
    flux_corr = flux / ext.extinguish(lam, Ebv=ebv)

    return flux_corr

def list_flat(old_list, new_list):
    for item in old_list:
        if type(item) == float:
            new_list.append(item)
        elif type(item) == int:
            new_list.append(item)
        elif type(item) == list:
            for item2 in item:
                new_list.append(item2)
    return new_list


def clean_images_velo(img, pa, a, b, img_err=None,SNR=3, n_re=2):
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
                    new_img = [img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]]
                    new_img = np.nanmedian(new_img)
                    if np.isnan(new_img):
                        img[i,j]=np.nan
                    else:
                        img[i, j] = new_img
    return img


def clean_images_flux(img, pa, a, b, img_err=None, SNR=3,n_re=2):
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
                img[i, j] = np.nan
    return img

def aperture_photometry(map, pa, a, b, n_re=2):
    flux = 0
    map[np.isnan(map)] = 0
    map[np.isinf(map)] = 0
    pa = pa - 90
    pa = np.radians(pa)
    y0, x0 = map.shape
    y0, x0 = y0 / 2, x0 / 2
    for i in range(len(map[:, 0])):
        for j in range(len(map[0, :])):
            side1 = (((j - x0) * np.cos(pa)) + ((i - y0) * np.sin(pa))) ** 2 / (a ** 2)
            side2 = (((j - x0) * np.sin(pa)) - ((i - y0) * np.cos(pa))) ** 2 / (b ** 2)
            if side1 + side2 < n_re**2:
                flux += map[i, j]
            else:
                map[i, j] = np.nan

    return flux


def BPT_pixels(HA, NII, OI, OIII, HB, SII, pa, a, b, output_file):
    BPT_map = np.ones(np.shape(HA))
    for m in [HA, HB, NII, SII, OI, OIII]:
        m[np.isnan(m)] = 0
        m[np.isinf(m)] = 0
    y0, x0 = HA.shape
    y0, x0 = y0 / 2, x0 / 2
    pa = pa - 90
    pa = np.radians(pa)
    for i in range(len(HA[:, 0])):
        for j in range(len(HA[0, :])):
            side1 = (((j - x0) * np.cos(pa)) + ((i - y0) * np.sin(pa))) ** 2 / (a ** 2)
            side2 = (((j - x0) * np.sin(pa)) - ((i - y0) * np.cos(pa))) ** 2 / (b ** 2)
            if side1 + side2 < 8:
                if np.isinf(-1 * OIII[i, j] / HB[i, j]) or np.isinf(-1 * SII[i, j] / HA[i, j]) or np.isnan(
                        OIII[i, j] / HB[i, j]) or np.isnan(SII[i, j] / HA[i, j]) \
                        or np.isinf(OIII[i, j] / HB[i, j]) or np.isinf(SII[i, j] / HA[i, j]):
                    BPT_map[i, j] = np.nan
                    return
                if np.log10(OIII[i, j] / HB[i, j]) > 1.30 + 0.72 / (
                        np.log10(SII[i, j] / HA[i, j]) - 0.32) and np.log10(
                    OIII[i, j] / HB[i, j]) < 1.89 * np.log10(SII[i, j] / HA[i, j]) + 0.76:
                    BPT_map[i, j] = 3
                    return
                if np.log10(OIII[i, j] / HB[i, j]) > 1.30 + 0.72 / (
                        np.log10(SII[i, j] / HA[i, j]) - 0.32) and np.log10(
                    OIII[i, j] / HB[i, j]) > 1.89 * np.log10(SII[i, j] / HA[i, j]) + 0.76:
                    BPT_map[i, j] = 2
                    return
                else:
                    pass
            else:
                BPT_map[i, j] = np.nan

    xx = np.arange(-4, 0.25, 0.01)
    yy = 1.19 + (0.61 / (xx - 0.47))
    nn = np.arange(-3, 0.03, 0.01)
    mm = 1.19 + (0.61 / (nn - 0.05))
    vv = 1.30 + (0.72 / (xx - 0.32))
    dd = 1.19 + (0.61 / (xx - 0.47))
    cc = np.arange(-0.31, 2, 0.01)
    pp = 1.89 * cc + 0.76
    hh = np.arange(-0.58, 2, 0.01)
    ll = 1.18 * hh + 1.30
    jj = np.arange(-0.19, 2, 0.01)
    kk = 1.01 * jj + 0.48

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey="row", figsize=(10, 5))
    plt.rcParams.update({"font.size": 12})
    ax1.scatter(np.log10(NII / HA), np.log10(OIII / HB), s=2, c="k")
    ax1.plot(xx, yy, color="k")
    ax1.plot(nn, mm, color="k", ls="dashdot")
    ax1.plot(jj, kk, color="k")
    ax2.scatter(np.log10(SII / HA), np.log10(OIII / HB), s=2, c="k")
    ax2.plot(xx, vv, color="k")
    ax2.plot(cc, pp, color="k")
    ax3.scatter(np.log10(OI / HA), np.log10(OIII / HB), s=2, c="k")
    ax3.plot(xx, dd, color="k")
    ax3.plot(hh, ll, color="k")
    for ax in [ax1, ax2, ax3]:
        ax.text(x=0.5, y=-0.5, s="LINERs")
        ax.text(x=-2, y=1.2, s="Seyferts")
        ax.text(x=-2.2, y=-1.0, s="Star Forming")
        ax.set_ylim(-1.3, 1.8)
        ax.set_xlim(-2.8, 1.33)
    ax1.set_xlabel(r"$\log$ [NII]/H$\alpha$")
    ax2.set_xlabel(r"$\log$ [SII]/H$\alpha$")
    ax3.set_xlabel(r"$\log$ [OI]/H$\alpha$")
    ax1.set_ylabel(r"$\log$ [OIII]/H$\beta$")
    plt.subplots_adjust(wspace=0)
    plt.savefig(output_file + "_BPT_pixels.pdf", bbox_inches="tight")

    return BPT_map


def BPT_plots(output_file, sample_file, n_re):
    HA_fluxes = []
    HB_fluxes = []
    OI_fluxes = []
    OIII_fluxes = []
    SII_fluxes = []
    NII_fluxes = []
    SFR = []
    re = []
    re_DL = []
    HA_err_fluxes = []
    SFR_err = []
    sample = pd.read_csv(sample_file)
    galaxies = sample["MAGPIID"].to_numpy()
    galaxies = galaxies[~np.isnan(galaxies)]
    galaxies = galaxies.astype(int)
    for g in galaxies:
        print("Beginning MAGPI" + str(g) + "...")
        csv_file = pd.read_csv(
            "MAGPI_Emission_Lines/MAGPI" + str(g)[:4] + "/MAGPI" + str(g)[:4] + "_source_catalogue.csv", skiprows=16)
        csv_file = csv_file[csv_file["MAGPIID"].isin([g])]
        z = csv_file["z"].to_numpy()[0]
        r50 = csv_file["R50_it"].to_numpy()[0] / 0.2
        q = csv_file["axrat_it"].to_numpy()[0]
        pa = csv_file["ang_it"].to_numpy()[0]
        DL = cosmo.luminosity_distance(z).to(u.kpc).value
        file = "MAGPI_Emission_Lines/MAGPI" + str(g)[:4] + "/MAGPI" + str(g)[
                                                                      :4] + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
            g) + "_GIST_EmissionLines.fits"
        if os.path.exists(file):
            pass
        else:
            print("No gas spectra!")
            HA_fluxes.append(np.nan)
            HA_err_fluxes.append(np.nan)
            HB_fluxes.append(np.nan)
            OIII_fluxes.append(np.nan)
            NII_fluxes.append(np.nan)
            OI_fluxes.append(np.nan)
            SII_fluxes.append(np.nan)
            SFR.append(np.nan)
            SFR_err.append(np.nan)
            re.append(r50)
            re_DL.append(np.radians(r50 / 3600) * cosmo.angular_diameter_distance(z).to(u.kpc))
            continue
        file = "MAGPI_Emission_Lines/MAGPI" + str(g)[:4] + "/MAGPI" + str(g)[
                                                                      :4] + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
            g) + "_GIST_EmissionLines.fits"
        fits_file = fits.open(file)
        flux_Ha = fits_file[49].data
        flux_Hb = fits_file[37].data
        flux_Ha_err = fits_file[50].data
        flux_Hb_err = fits_file[38].data
        OIII = fits_file[39].data
        OIII_err = fits_file[30].data
        NII = fits_file[51].data
        NII_err = fits_file[52].data
        OI = fits_file[47].data
        OI_err = fits_file[48].data
        SII = fits_file[53].data
        SII = SII + fits_file[55].data
        SII_err = fits_file[54].data
        SII_err = SII_err + fits_file[56].data
        fits_file.close()

        HA = clean_images_flux(flux_Ha, pa, r50, r50 * q, img_err=flux_Ha / flux_Ha_err)
        HA_err = clean_images_flux(flux_Ha_err, pa, r50, r50 * q)
        HB = clean_images_flux(flux_Hb, pa, r50, r50 * q, img_err=flux_Hb / flux_Hb_err)
        OI = clean_images_flux(OI, pa, r50, r50 * q, img_err=OI / OI_err)
        OIII = clean_images_flux(OIII, pa, r50, r50 * q, img_err=OIII / OIII_err)
        NII = clean_images_flux(NII, pa, r50, r50 * q, img_err=NII / NII_err)
        SII = clean_images_flux(SII, pa, r50, r50 * q, img_err=SII / SII_err)

        if os.path.exists("/Volumes/DS/MAGPI/MAGPI_Plots/plots/MAGPI" + str(g)[:4] + "/BPT_plots"):
            shutil.rmtree("/Volumes/DS/MAGPI/MAGPI_Plots/plots/MAGPI" + str(g)[:4] + "/BPT_plots")
        os.mkdir("/Volumes/DS/MAGPI/MAGPI_Plots/plots/MAGPI" + str(g)[:4] + "/BPT_plots")

        bpt_map = BPT_pixels(HA, NII, OI, OIII, HB, SII, pa, r50, r50 * q,
                             "/Volumes/DS/MAGPI/MAGPI_Plots/plots/MAGPI" + str(g)[:4] + "/BPT_plots/" + str(g))

        HA = clean_images_flux(flux_Ha, pa, r50, r50 * q, img_err=flux_Ha / flux_Ha_err,n_re=n_re)
        HA_err = clean_images_flux(flux_Ha_err, pa, r50, r50 * q)
        HB = clean_images_flux(flux_Hb, pa, r50, r50 * q, img_err=flux_Hb / flux_Hb_err,n_re=n_re)
        OI = clean_images_flux(OI, pa, r50, r50 * q, img_err=OI / OI_err)
        OIII = clean_images_flux(OIII, pa, r50, r50 * q, img_err=OIII / OIII_err,n_re=n_re)
        NII = clean_images_flux(NII, pa, r50, r50 * q, img_err=NII / NII_err,n_re=n_re)
        SII = clean_images_flux(SII, pa, r50, r50 * q, img_err=SII / SII_err,n_re=n_re)

        fig, ax = plt.subplots()
        ax.imshow(flux_Ha,origin="lower")
        plt.savefig("/Volumes/DS/MAGPI/MAGPI_Plots/plots/MAGPI" + str(g)[:4] + "/BPT_plots/" + str(g) + "check.pdf",
                    bbox_inches='tight')

        if not bpt_map == None:
            fig, ax = plt.subplots()
            p = ax.imshow(bpt_map)
            cbar = plt.colorbar(p, ax=ax, ticks=[1, 2, 3])
            cbar.ax.set_yticklabels(["HII", "Seyfert", "LINER"])
            plt.savefig("/Volumes/DS/MAGPI/MAGPI_Plots/plots/MAGPI" + str(g)[:4] + "/BPT_plots/" + str(g) + "bpt_map.pdf")

        # HA_flux = aperture_photometry(HA,pa,r50,r50 * q,n_re)
        # HA_err_flux = aperture_photometry(HA_err, pa,r50,r50 * q,n_re)
        # HB_flux = aperture_photometry(HB, pa,r50,r50 * q,n_re)
        # OIII_flux = aperture_photometry(OIII, pa,r50,r50 * q,n_re)
        # NII_flux = aperture_photometry(NII,pa,r50,r50 * q,n_re)
        # OI_flux = aperture_photometry(OI, pa,r50,r50 * q,n_re)
        # SII_flux = aperture_photometry(SII, pa,r50,r50 * q,n_re)
        HA_flux = np.nansum(HA)
        HA_err_flux = np.nansum(HA_err)
        HB_flux = np.nansum(HB)
        OIII_flux = np.nansum(OIII)
        NII_flux = np.nansum(NII)
        OI_flux = np.nansum(OI)
        SII_flux = np.nansum(SII)

        DL = cosmo.luminosity_distance(z).to(u.cm).value
        balmer = HA_flux / HB_flux
        HA_flux_corr = dust_corr(HA_flux * 1e-20, balmer, 6562 * u.AA, CCM89)
        HA_err_flux = dust_corr(HA_err_flux * 1e-20, balmer, 6562 * u.AA, CCM89)
        lum = HA_flux_corr * (4 * np.pi * DL ** 2)
        lum_err = HA_err_flux * (4 * np.pi * DL ** 2)
        SFR.append(lum * 5.5e-42)
        SFR_err.append(lum_err * 5.5e-42)

        HA_fluxes.append(HA_flux)
        HA_err_fluxes.append(HA_err_flux)
        HB_fluxes.append(HB_flux)
        OIII_fluxes.append(OIII_flux)
        NII_fluxes.append(NII_flux)
        OI_fluxes.append(OI_flux)
        SII_fluxes.append(SII_flux)
        re.append(r50)
        re_DL.append(np.radians(r50 / 3600) * cosmo.angular_diameter_distance(z).to(u.kpc))

    HA_fluxes = np.array(HA_fluxes)
    HA_err_fluxes = np.array(HA_err_fluxes)
    HB_fluxes = np.array(HB_fluxes)
    OIII_fluxes = np.array(OIII_fluxes)
    NII_fluxes = np.array(NII_fluxes)
    OI_fluxes = np.array(OI_fluxes)
    SII_fluxes = np.array(SII_fluxes)
    SFR = np.array(SFR)
    SFR_err = np.array(SFR_err)
    re = np.array(re)
    re_DL = np.array(re_DL)

    sf_sy_ln = np.zeros(len(HA_fluxes))
    for i in range(len(HA_fluxes)):
        if np.log10(OI_fluxes[i] / HA_fluxes[i]) > -0.59 and np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.18 * np.log10(
                OI_fluxes[i] / HA_fluxes[i]) + 1.30:
            print(galaxies[i], "LINER!")
            sf_sy_ln[i] = 3
            pass
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.19 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.47)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and \
                np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.33 + (
                0.73 / (np.log10(OI_fluxes[i] / HA_fluxes[i]) + 0.59)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) < 1.89 * np.log10(SII_fluxes[i] / HA_fluxes[i]) + 0.76:
            print(galaxies[i], "LINER!")
            sf_sy_ln[i] = 3
            pass
        if np.log10(OI_fluxes[i] / HA_fluxes[i]) > -0.59 and np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.89 * np.log10(
                SII_fluxes[i] / HA_fluxes[i]) + 0.76 and np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.18 * np.log10(
            OI_fluxes[i] / HA_fluxes[i]) + 1.30:
            print(galaxies[i],"Seyfert!")
            sf_sy_ln[i] = 2
            pass
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.19 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.47)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and \
                np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.33 + (
                0.73 / (np.log10(OI_fluxes[i] / HA_fluxes[i]) + 0.59)):
            print(galaxies[i], "Seyfert!")
            sf_sy_ln[i] = 2
            continue
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.05)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + (0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32)) and \
                np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.33 + (
                0.73 / (np.log10(OI_fluxes[i] / HA_fluxes[i]) - +0.59)):
            print(galaxies[i], "Star Forming!")
            sf_sy_ln[i] = 1
            pass
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.05)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.19 + 0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.47):
            print(galaxies[i], "Comp!")
            sf_sy_ln[i] = 0
            pass
    print("[SII]-BPT")
    SII_bpt = np.zeros(len(HA_fluxes))
    for i in range(len(HA_fluxes)):
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (
                np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) < 1.89 * np.log10(SII_fluxes[i] / HA_fluxes[i]) + 0.76:
            SII_bpt[i] = 3
            #print(galaxies[i], "LINER!")
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (
                np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.89 * np.log10(SII_fluxes[i] / HA_fluxes[i]) + 0.76:
            SII_bpt[i] = 2
            #print(galaxies[i], "Seyfert!")
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + 0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32):
            SII_bpt[i] = 1
            #print(galaxies[i], "Star Forming!")
    for k,h in zip(SII_bpt,sf_sy_ln):
        if k==2 and h==2:
            print("Both saying Seyfert")
        if k==3 and h==3:
            print("Both saying LINER")
    print("All Done!")

    print(len(galaxies),len(HA_fluxes),len(HA_err_fluxes),len(HB_fluxes),len(OI_fluxes),len(OIII_fluxes),len(NII_fluxes),len(SII_fluxes),len(sf_sy_ln),len(SII_bpt),len(SFR),len(SFR_err),len(re))
    df = pd.DataFrame({"MAGPIID": galaxies,
                       "Ha": HA_fluxes,
                       "Ha_err": HA_err_fluxes,
                       "Hb": HB_fluxes,
                       "[OI]6302": OI_fluxes,
                       "[OIII]5008": OIII_fluxes,
                       "[NII]6585": NII_fluxes,
                       "[SII]6718": SII_fluxes,
                       "type(sf+AGN=0, sf=1, sy=2, ln=3)": sf_sy_ln,
                       "type(sf=1, sy=2, ln=3) SII": SII_bpt,
                       "SFR": SFR,
                       "SFR_err": SFR_err,
                       "re, arcsec": re * 0.2,
                       "re, kpc": re_DL})
    df.to_csv(output_file, index=False)

    xx = np.arange(-4, 0.25, 0.01)
    yy = 1.19 + (0.61 / (xx - 0.47))
    nn = np.arange(-3, 0.03, 0.01)
    mm = 1.19 + (0.61 / (nn - 0.05))
    vv = 1.30 + (0.72 / (xx - 0.32))
    dd = 1.19 + (0.61 / (xx - 0.47))
    cc = np.arange(-0.31, 2, 0.01)
    pp = 1.89 * cc + 0.76
    hh = np.arange(-0.58, 2, 0.01)
    ll = 1.18 * hh + 1.30
    jj = np.arange(-0.19, 2, 0.01)
    kk = 1.01 * jj + 0.48

    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey="row", figsize=(10, 3))
    ax1.scatter(np.log10(NII_fluxes / HA_fluxes), np.log10(OIII_fluxes / HB_fluxes), s=2, c="k")
    ax1.plot(xx, yy, color="k")
    ax1.plot(nn, mm, color="k", ls="dashdot")
    ax1.plot(jj, kk, color="k")
    ax2.scatter(np.log10(SII_fluxes / HA_fluxes), np.log10(OIII_fluxes / HB_fluxes), s=2, c="k")
    ax2.plot(xx, vv, color="k")
    ax2.plot(cc, pp, color="k")
    ax3.scatter(np.log10(OI_fluxes / HA_fluxes), np.log10(OIII_fluxes / HB_fluxes), s=2, c="k")
    ax3.plot(xx, dd, color="k")
    ax3.plot(hh, ll, color="k")
    for ax in [ax1, ax2, ax3]:
        ax.text(x=0.3, y=-0.5, s="LINERs")
        ax.text(x=-2, y=1.2, s="Seyferts")
        ax.text(x=-2.2, y=-1.0, s="Star Forming")
        ax.set_ylim(-1.3, 1.8)
        ax.set_xlim(-2.8, 1.33)
    ax1.set_xlabel(r"$\log$ [NII]/H$\alpha$")
    ax2.set_xlabel(r"$\log$ [SII]/H$\alpha$")
    ax3.set_xlabel(r"$\log$ [OI]/H$\alpha$")
    ax1.set_ylabel(r"$\log$ [OIII]/H$\beta$")
    plt.subplots_adjust(wspace=0)
    plt.savefig("MAGPI_Plots/plots/BPT.pdf", bbox_inches="tight")

    return HA_fluxes, NII_fluxes, OIII_fluxes, HB_fluxes, SII_fluxes, OI_fluxes

def stellar_gas_plots(galaxy, n_ells=3, SNR_star=3, SNR_gas=20):
    field_name = str(galaxy)[:4]
    csv_file = pd.read_csv(
        "MAGPI_Emission_Lines/MAGPI" + str(field_name) + "/MAGPI" + str(field_name) + "_source_catalogue.csv",
        skiprows=16)
    csv_file = csv_file[csv_file['MAGPIID'].isin([galaxy])]
    z = csv_file["z"].to_numpy()[0]
    r50 = csv_file["R50_it"].to_numpy()[0] / 0.2
    q = csv_file["axrat_it"].to_numpy()[0]
    pa = csv_file["ang_it"].to_numpy()[0]
    DL = cosmo.luminosity_distance(z).to(u.kpc).value
    pix = np.radians(0.33 / 3600) * DL

    star_file = "MAGPI_Absorption_Lines/MAGPI" + field_name + "/galaxies/" + str(
        galaxy) + "_kinematics_ppxf-maps.fits"
    gas_file = "MAGPI_Emission_Lines/MAGPI" + field_name + "/MAGPI" + field_name + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
        galaxy) + "_GIST_EmissionLines.fits"

    if os.path.exists(gas_file) and os.path.exists(star_file):
        print("Has gas and stellar kinematics!")
        starfile = fits.open(star_file)
        gasfile = fits.open(gas_file)
        s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
        s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_err=s_flux)
        s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_err=s_flux)
        s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_err=s_flux)

        g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
            gasfile[10].data, gasfile[11].data

        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err

        clip = np.nanmax(s_flux)
        y0, x0 = s_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Stellar SNR = {clip:.2f}...")
        if clip < SNR_star:
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its heinous looking")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its heinous looking")
            return
        ha_check = np.count_nonzero(~np.isnan(g_flux))
        if ha_check < 50:
            print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " Ha spaxels survive!")
            print("Finding Brightest Line")
            max_line = pd.read_csv("MAGPI_csv/MAGPI_Emission_Max_Line.csv")
            max_line = max_line[max_line["MAGPIID"].isin([galaxy])]
            bright_line = max_line["MAX_LINE"].to_numpy()[0]
            print("Brightest line is " + bright_line)
            bright_line_err = max_line["MAX_LINE"].to_numpy()[0]

            g_velo = gasfile[9].data
            g_flux = gasfile[bright_line].data
            g_flux_err = gasfile[bright_line_err].data
            g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_flux = clean_images_velo(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)

            print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " spaxels survive!")
            bl_check = np.count_nonzero(~np.isnan(g_flux))
            if ha_check < bl_check:
                print(bright_line + " is better!")
            else:
                print(f"Ha is better")
                g_velo = gasfile[9].data
                g_flux = gasfile[49].data
                g_flux_err = gasfile[50].data
                g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
                g_flux = clean_images_velo(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50 + step
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                return

            print("Doing kinemetry on stars and gas!")

            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                           cover=0.95)
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                           cover=0.95)
            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            pa_g = kg.pa[-1]
            pa_s = ks.pa[-1]

            fig, ax = plt.subplots()
            ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
            ax.plot(ks.rad, ks1, zorder=1)
            ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
            ax.plot(kg.rad, kg1, zorder=1)
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_Vrot.pdf",
                        bbox_inches="tight")

            starfile.close()
            gasfile.close()
            starfile = fits.open(star_file)
            gasfile = fits.open(gas_file)

            if ha_check > bl_check:
                s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[
                    4].data
                s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_err=s_flux)
                s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_err=s_flux)
                s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_err=s_flux)

                g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                gasfile[10].data, gasfile[11].data
                g_velo = clean_images_velo(g_velo, pa, r50, r50 * q)
                g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q)
                g_flux = clean_images_flux(g_flux, pa, r50, r50 * q)
                g_flux = g_flux / g_flux_err
                starfile.close()
                gasfile.close()

                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))
                p1 = ax1.imshow(s_flux, origin="lower")
                p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-0.5 * np.nanmax(s_velo),
                                vmax=0.5 * np.nanmax(s_velo))
                p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
                p4 = ax4.imshow(g_flux, origin="lower")
                p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                                vmax=0.9 * np.nanmax(g_velo))
                p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(g_sigma))
                ax1.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
                ax4.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
                ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                      height=2 * r50 / q, angle=pa_s, fc="none", ec="magenta"))
                ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                      height=2 * r50 / q, angle=pa_g, fc="none", ec="magenta"))
                ax1.set_ylabel("Stars")
                ax4.set_ylabel("Gas")
                for p, ax, label in zip([p1, p2, p3, p4, p5, p6], [ax1, ax2, ax3, ax4, ax5, ax6],
                                        [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]", r"SNR [H$\alpha$]",
                                         r"V [kms$^{-1}$]",
                                         r"$\sigma$ [kms$^{-1}$]"]):
                    plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                print("plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf")
                plt.savefig("MAGPI_Plots/plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf", bbox_inches="tight")
                plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_fluxplots.pdf",
                            bbox_inches="tight")

                hdr = fits.Header()
                hdr["COMMENT"] = '========================================================================'
                hdr["COMMENT"] = 'This FITS file contains the gas SNR map [1], gas observed velocity field'
                hdr["COMMENT"] = '[2], simple gas circular velocity model [3], gas kinemetry model'
                hdr["COMMENT"] = '(with higher order terms), stars SNR map [4], gas velocity residual map '
                hdr["COMMENT"] = ' [5],stars SNR map [6], star velocity model [7], star simple circular ve'
                hdr["COMMENT"] = 'locity model [8], stars kinemetry model [9], star velocity residual map'
                hdr["COMMENT"] = ' [10]'
                hdr["COMMENT"] = '========================================================================'
                hdr["OBJECT"] = str(galaxy)
                n = None
                hdu0 = fits.PrimaryHDU(n, header=hdr)
                hdu1 = fits.ImageHDU(g_flux, name="SNR_Stars", header=hdr)
                hdu2 = fits.ImageHDU(g_velo, name="Data", header=hdr)
                hdu3 = fits.ImageHDU(kg.velcirc, name="Velcirc", header=hdr)
                hdu4 = fits.ImageHDU(kg.velkin, name="VelKin", header=hdr)
                hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
                hdu6 = fits.ImageHDU(s_flux, name="SNR_Stars", header=hdr)
                hdu7 = fits.ImageHDU(s_velo, name="Data", header=hdr)
                hdu8 = fits.ImageHDU(ks.velcirc, name="Velcirc", header=hdr)
                hdu9 = fits.ImageHDU(ks.velkin, name="VelKin", header=hdr)
                hdu10 = fits.ImageHDU(s_velo - ks.velcirc, name="V - VelKin", header=hdr)
                hdr["BUNIT"] = None

                out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8, hdu9, hdu10])
                out.writeto("MAGPI_Plots/plots/MAGPI" + field_name + "/fits_files/" + str(galaxy) + "_stellar_kinemetry.fits",
                            overwrite=True)
            else:
                s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, \
                    starfile[4].data
                s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_err=s_flux)
                s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_err=s_flux)
                s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_err=s_flux)

                g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[bright_line].data, gasfile[
                    bright_line_err].data, gasfile[
                    9].data, gasfile[10].data, gasfile[11].data
                g_velo = clean_images_velo(g_velo, pa, r50, r50 * q)
                g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q)
                g_flux = clean_images_flux(g_flux, pa, r50, r50 * q)

                starfile.close()
                gasfile.close()

                fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))
                p1 = ax1.imshow(s_flux, origin="lower")
                p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-0.5 * np.nanmax(s_velo),
                                vmax=0.5 * np.nanmax(s_velo))
                p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
                p4 = ax4.imshow(g_flux, origin="lower")
                p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                                vmax=0.9 * np.nanmax(g_velo))
                p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(s_sigma))
                ax1.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
                ax4.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
                ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                      height=2 * r50 / q, angle=pa_s, fc="none", ec="magenta"))
                ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                      height=2 * r50 / q, angle=pa_g, fc="none", ec="magenta"))
                ax1.set_ylabel("Stars")
                ax4.set_ylabel("Gas")
                for p, ax, label in zip([p1, p2, p3, p4, p5, p6], [ax1, ax2, ax3, ax4, ax5, ax6],
                                        [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]",
                                         bright_line[:-2] + " [x10$^{-20}$ erg s$^{-1}$ cm$^{-2}$]",
                                         r"V [kms$^{-1}$]",
                                         r"$\sigma$ [kms$^{-1}$]"]):
                    plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                plt.savefig("MAGPI_Plots/plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf", bbox_inches="tight")
                plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_fluxplots.pdf",
                            bbox_inches="tight")

                hdr = fits.Header()
                hdr["COMMENT"] = '========================================================================'
                hdr["COMMENT"] = 'This FITS file contains the gas SNR map [1], gas observed velocity field'
                hdr["COMMENT"] = '[2], simple gas circular velocity model [3], gas kinemetry model'
                hdr["COMMENT"] = '(with higher order terms), stars SNR map [4], gas velocity residual map '
                hdr["COMMENT"] = ' [5],stars SNR map [6], star velocity model [7], star simple circular ve'
                hdr["COMMENT"] = 'locity model [8], stars kinemetry model [9], star velocity residual map'
                hdr["COMMENT"] = ' [10]'
                hdr["COMMENT"] = '========================================================================'
                hdr["OBJECT"] = str(galaxy)
                n = None
                hdu0 = fits.PrimaryHDU(n, header=hdr)
                hdu1 = fits.ImageHDU(g_flux, name="SNR_Stars", header=hdr)
                hdu2 = fits.ImageHDU(g_velo, name="Gas Velo", header=hdr)
                hdu3 = fits.ImageHDU(kg.velcirc, name="Gas Velcirc", header=hdr)
                hdu4 = fits.ImageHDU(kg.velkin, name="Gas VelKin", header=hdr)
                hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
                hdu6 = fits.ImageHDU(s_flux, name="SNR_Stars", header=hdr)
                hdu7 = fits.ImageHDU(s_velo, name="Stars Velo", header=hdr)
                hdu8 = fits.ImageHDU(ks.velcirc, name="Stars Velcirc", header=hdr)
                hdu9 = fits.ImageHDU(ks.velkin, name="Stars VelKin", header=hdr)
                hdu10 = fits.ImageHDU(s_velo - ks.velcirc, name="V - VelKin", header=hdr)
                hdr["BUNIT"] = None

                out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8, hdu9, hdu10])
                out.writeto("MAGPI_Plots/plots/MAGPI" + field_name + "/fits_files/" + str(galaxy) + "_stellar_kinemetry.fits",
                            overwrite=True)


        else:
            print("Doing kinemetry on stars and gas!")
            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50 + step
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                return

            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                           cover=0.95)
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                           cover=0.95)

            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            pa_g = kg.pa[-1]
            pa_s = ks.pa[-1]

            fig, ax = plt.subplots()
            ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
            ax.plot(ks.rad, ks1, zorder=1)
            ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
            ax.plot(kg.rad, kg1, zorder=1)
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_Vrot.pdf",
                        bbox_inches="tight")

            starfile.close()
            gasfile.close()
            starfile = fits.open(star_file)
            gasfile = fits.open(gas_file)

            s_flux, s_velo, s_sigma = starfile[7].data, starfile[1].data, starfile[4].data
            g_flux, g_flux_err, g_velo, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[
                11].data
            s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_err=s_flux)
            s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_err=s_flux)
            s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_err=s_flux)

            g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            starfile.close()
            gasfile.close()

            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))
            p1 = ax1.imshow(s_flux, origin="lower")
            p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-220, vmax=220)
            p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
            p4 = ax4.imshow(g_flux, origin="lower")
            p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                            vmax=0.9 * np.nanmax(g_velo))
            p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(s_sigma))
            ax1.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
            ax4.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
            ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                  height=2 * r50 / q, angle=pa_s, fc="none", ec="magenta"))
            ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                  height=2 * r50 / q, angle=pa_g, fc="none", ec="magenta"))
            ax1.set_ylabel("Stars")
            ax4.set_ylabel("Gas")
            for p, ax, label in zip([p1, p2, p3, p4, p5, p6], [ax1, ax2, ax3, ax4, ax5, ax6],
                                    [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]", r"SNR [H$\alpha$]",
                                     r"V [kms$^{-1}$]",
                                     r"$\sigma$ [kms$^{-1}$]"]):
                plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
            plt.savefig("MAGPI_Plots/plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf", bbox_inches="tight")
            plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_fluxplots.pdf",
                        bbox_inches="tight")

            hdr = fits.Header()
            hdr["COMMENT"] = '========================================================================'
            hdr["COMMENT"] = 'This FITS file contains the gas SNR map [1], gas observed velocity field'
            hdr["COMMENT"] = '[2], simple gas circular velocity model [3], gas kinemetry model'
            hdr["COMMENT"] = '(with higher order terms), stars SNR map [4], gas velocity residual map '
            hdr["COMMENT"] = ' [5],stars SNR map [6], star velocity model [7], star simple circular ve'
            hdr["COMMENT"] = 'locity model [8], stars kinemetry model [9], star velocity residual map'
            hdr["COMMENT"] = ' [10]'
            hdr["COMMENT"] = '========================================================================'
            hdr["OBJECT"] = str(galaxy)
            n = None
            hdu0 = fits.PrimaryHDU(n, header=hdr)
            hdu1 = fits.ImageHDU(g_flux, name="SNR_Gas", header=hdr)
            hdu2 = fits.ImageHDU(g_velo, name="Data", header=hdr)
            hdu3 = fits.ImageHDU(kg.velcirc, name="Velcirc", header=hdr)
            hdu4 = fits.ImageHDU(kg.velkin, name="VelKin", header=hdr)
            hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
            hdu6 = fits.ImageHDU(s_flux, name="SNR_Stars", header=hdr)
            hdu7 = fits.ImageHDU(s_velo, name="Data", header=hdr)
            hdu8 = fits.ImageHDU(ks.velcirc, name="Velcirc", header=hdr)
            hdu9 = fits.ImageHDU(ks.velkin, name="VelKin", header=hdr)
            hdu10 = fits.ImageHDU(s_velo - ks.velcirc, name="V - VelKin", header=hdr)
            hdr["BUNIT"] = None

            out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6, hdu7, hdu8, hdu9, hdu10])
            out.writeto("MAGPI_Plots/plots/MAGPI" + field_name + "/fits_files/" + str(galaxy) + "_stellar_kinemetry.fits",
                        overwrite=True)


    elif os.path.exists(gas_file) and os.path.exists(star_file) == False:
        print("Has gas kinematics but no stars!")
        # log_file.write("Has gas kinematics but no stars!\n")
        gasfile = fits.open(gas_file)
        g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[
            10].data, gasfile[11].data
        g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_velo_err = clean_images_velo(g_velo_err, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
        g_flux = g_flux / g_flux_err

        clip = np.nanmax(g_flux)
        y0, x0 = g_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Gas SNR = {clip:.2f}...")
        if clip < SNR_gas:
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its heinous looking")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        ha_check = np.count_nonzero(~np.isnan(g_flux))
        if ha_check < 50:
            print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " Ha spaxels survive!")
            print("Finding Brightest Line")
            max_line = pd.read_csv("MAGPI_csv/MAGPI_Emission_Max_Line.csv")
            max_line = max_line[max_line["MAGPIID"].isin([galaxy])]
            bright_line = max_line["MAX_LINE"].to_numpy()[0]
            print("Brightest line is " + bright_line)
            bright_line_err = max_line["MAX_LINE"].to_numpy()[0]

            g_velo = gasfile[9].data
            g_flux = gasfile[bright_line].data
            g_flux_err = gasfile[bright_line_err].data
            g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
            g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)

            print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " spaxels survive!")
            bl_check = np.count_nonzero(~np.isnan(g_flux))
            if ha_check < bl_check:
                print(bright_line + " is better!")
            else:
                print(f"Ha is better")
                g_velo = gasfile[9].data
                g_flux = gasfile[49].data
                g_flux_err = gasfile[50].data
                g_velo = clean_images_velo(g_velo, pa, r50, r50 * q, img_err=g_flux / g_flux_err)
                g_flux = clean_images_flux(g_flux, pa, r50, r50 * q, img_err=g_flux / g_flux_err)

            step = (0.65 / 2) / 0.2
            start = (0.65 / 2) / 0.2 - step
            end = 1 * r50 + step
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print(f"{len(rad)} ellipse/s, Not enough ellipses!")
                return
            kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                           cover=0.95)

            kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
            pa_g = kg.pa[-1]
            q_g = kg.q[-1]

            fig, ax = plt.subplots()
            ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
            ax.plot(kg.rad, kg1, zorder=1)
            ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
            ax.set_xlabel("R [pix]")
            ax.legend()
            plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_Vrot.pdf",
                        bbox_inches="tight")

            gasfile.close()
            gasfile = fits.open(gas_file)
            if ha_check > bl_check:
                g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[
                    9].data, gasfile[10].data, gasfile[11].data
                g_velo = clean_images_velo(g_velo, pa, r50, r50 * q)
                g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q)
                g_flux = clean_images_flux(g_flux, pa, r50, r50 * q)
                g_flux = g_flux / g_flux_err
                gasfile.close()

                fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(5, 8))
                p4 = ax4.imshow(g_flux, origin="lower")
                p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                                vmax=0.9 * np.nanmax(g_velo))
                p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(g_sigma))
                ax4.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
                ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                      height=2 * r50 / q, angle=pa_g, fc="none", ec="magenta"))
                ax4.set_ylabel("Gas")
                for p, ax, label in zip([p4, p5, p6], [ax4, ax5, ax6],
                                        [r"SNR [H$_\alpha$]",
                                         r"V [kms$^{-1}$]",
                                         r"$\sigma$ [kms$^{-1}$]"]):
                    plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                plt.savefig("MAGPI_Plots/plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf", bbox_inches="tight")
                plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_fluxplots.pdf",
                            bbox_inches="tight")

                hdr = fits.Header()
                hdr["COMMENT"] = '========================================================================'
                hdr["COMMENT"] = 'This FITS file contains the gas SNR map [1], gas observed velocity field'
                hdr["COMMENT"] = '[2], simple gas circular velocity model [3], gas kinemetry model'
                hdr["COMMENT"] = '(with higher order terms), stars SNR map [4], gas velocity residual map '
                hdr["COMMENT"] = ' [5],stars SNR map [6], star velocity model [7], star simple circular ve'
                hdr["COMMENT"] = 'locity model [8], stars kinemetry model [9], star velocity residual map'
                hdr["COMMENT"] = ' [10]'
                hdr["COMMENT"] = '========================================================================'
                hdr["OBJECT"] = galaxy
                n = None
                hdu0 = fits.PrimaryHDU(n, header=hdr)
                hdu1 = fits.ImageHDU(g_flux, name="SNR_Gas", header=hdr)
                hdu2 = fits.ImageHDU(g_velo, name="Data", header=hdr)
                hdu3 = fits.ImageHDU(kg.velcirc, name="Velcirc", header=hdr)
                hdu4 = fits.ImageHDU(kg.velkin, name="VelKin", header=hdr)
                hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
                hdr["BUNIT"] = None

                out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
                out.writeto("MAGPI_Plots/plots/MAGPI" + field_name + "/fits_files/" + str(galaxy) + "_stellar_kinemetry.fits",
                            overwrite=True)

            else:
                g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[bright_line].data, gasfile[
                    bright_line_err].data, gasfile[
                    9].data, gasfile[10].data, gasfile[11].data
                g_velo = clean_images_velo(g_velo, pa, r50, r50 * q)
                g_sigma = clean_images_velo(g_sigma, pa, r50, r50 * q)
                g_flux = clean_images_flux(g_flux, pa, r50, r50 * q)
                gasfile.close()

                fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(5, 8))
                p4 = ax4.imshow(g_flux, origin="lower")
                p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                                vmax=0.9 * np.nanmax(g_velo))
                p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(g_sigma))
                ax4.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
                ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                                      height=2 * r50 / q, angle=pa_g, fc="none", ec="magenta"))
                ax4.set_ylabel("Gas")
                for p, ax, label in zip([p4, p5, p6], [ax4, ax5, ax6],
                                        [bright_line[:-2] + r" [x10$^{-20}$ erg s$^{-1}$ cm$^{-2}$]",
                                         r"V [kms$^{-1}$]",
                                         r"$\sigma$ [kms$^{-1}$]"]):
                    plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                plt.savefig("MAGPI_Plots/plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf", bbox_inches="tight")
                plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_fluxplots.pdf",
                            bbox_inches="tight")

                hdr = fits.Header()
                hdr["COMMENT"] = '========================================================================'
                hdr["COMMENT"] = 'This FITS file contains the gas SNR map [1], gas observed velocity field'
                hdr["COMMENT"] = '[2], simple gas circular velocity model [3], gas kinemetry model'
                hdr["COMMENT"] = '(with higher order terms), stars SNR map [4], gas velocity residual map '
                hdr["COMMENT"] = ' [5],stars SNR map [6], star velocity model [7], star simple circular ve'
                hdr["COMMENT"] = 'locity model [8], stars kinemetry model [9], star velocity residual map'
                hdr["COMMENT"] = ' [10]'
                hdr["COMMENT"] = '========================================================================'
                hdr["OBJECT"] = galaxy
                n = None
                hdu0 = fits.PrimaryHDU(n, header=hdr)
                hdu1 = fits.ImageHDU(g_flux, name="SNR_Gas", header=hdr)
                hdu2 = fits.ImageHDU(g_velo, name="Data", header=hdr)
                hdu3 = fits.ImageHDU(kg.velcirc, name="Velcirc", header=hdr)
                hdu4 = fits.ImageHDU(kg.velkin, name="VelKin", header=hdr)
                hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
                hdr["BUNIT"] = None

                out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
                out.writeto("MAGPI_Plots/plots/MAGPI" + field_name + "/fits_files/" + str(galaxy) + "_stellar_kinemetry.fits",
                            overwrite=True)


    elif os.path.exists(gas_file) == False and os.path.exists(star_file):
        print("Has stellar kinematics but no gas!")
        starfile = fits.open(star_file)
        s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
        s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_err=s_flux)
        s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_err=s_flux)
        s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_err=s_flux)

        clip = np.nanmax(s_flux)
        y0, x0 = s_flux.shape
        x0 = int(x0 / 2)
        y0 = int(y0 / 2)
        print(f"Max Stellar SNR = {clip:.2f}...")
        if clip < SNR_star:
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return
        elif np.isinf(clip) or np.isnan(clip):
            print("Not Plotting or doing Kinemetry on " + str(galaxy) + " because its heinous looking\n")
            return

        step = (0.65 / 2) / 0.2
        start = (0.65 / 2) / 0.2 - step
        end = 1 * r50 + step
        rad = np.arange(start, end, step)
        if len(rad) < n_ells:
            print(f"{len(rad)} ellipse/s, Not enough ellipses!")
            return
        ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                       bmodel=True, rangePA=[0, 360], rangeQ=[q - 0.1, q + 0.1], allterms=True,
                       cover=0.95)
        ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
        pa_s = ks.pa[-1]

        fig, ax = plt.subplots()
        ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
        ax.plot(ks.rad, ks1, zorder=1)
        ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
        ax.set_xlabel("R [pix]")
        ax.legend()
        plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_Vrot.pdf",
                    bbox_inches="tight")

        starfile.close()
        starfile = fits.open(star_file)
        s_flux, s_velo, s_sigma = starfile[7].data, starfile[1].data, starfile[4].data

        s_velo = clean_images_velo(s_velo, pa, r50, r50 * q, img_err=s_flux)
        s_velo_err = clean_images_velo(s_velo_err, pa, r50, r50 * q, img_err=s_flux)
        s_sigma = clean_images_velo(s_sigma, pa, r50, r50 * q, img_err=s_flux)

        starfile.close()

        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10, 8))
        p1 = ax1.imshow(s_flux, origin="lower")
        p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-220, vmax=220)
        p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
        ax1.add_patch(Circle(xy=(pix, pix), radius=pix, fc="none", ec="k"))
        ax1.add_patch(Ellipse(xy=(x0, y0), width=2 * r50,
                              height=2 * r50 / q, angle=pa_s, fc="none", ec="magenta"))
        ax1.set_ylabel("Stars")
        for p, ax, label in zip([p1, p2, p3], [ax1, ax2, ax3],
                                [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]", r"SNR",
                                 r"V [kms$^{-1}$]",
                                 r"$\sigma$ [kms$^{-1}$]"]):
            plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
        plt.savefig("MAGPI_Plots/plots/flux_velo_plots/" + str(galaxy) + "_fluxplots.pdf", bbox_inches="tight")
        plt.savefig("MAGPI_Plots/plots/MAGPI" + field_name + "/flux_plots/" + str(galaxy) + "_fluxplots.pdf",
                    bbox_inches="tight")

        hdr = fits.Header()
        hdr["COMMENT"] = '========================================================================'
        hdr["COMMENT"] = 'This FITS file contains the gas SNR map [1], gas observed velocity field'
        hdr["COMMENT"] = '[2], simple gas circular velocity model [3], gas kinemetry model'
        hdr["COMMENT"] = '(with higher order terms), stars SNR map [4], gas velocity residual map '
        hdr["COMMENT"] = ' [5],stars SNR map [6], star velocity model [7], star simple circular ve'
        hdr["COMMENT"] = 'locity model [8], stars kinemetry model [9], star velocity residual map'
        hdr["COMMENT"] = ' [10]'
        hdr["COMMENT"] = '========================================================================'
        hdr["OBJECT"] = str(galaxy)
        n = None
        hdu0 = fits.PrimaryHDU(n, header=hdr)
        hdu1 = fits.ImageHDU(s_flux, name="SNR_Stars", header=hdr)
        hdu2 = fits.ImageHDU(s_velo, name="Data", header=hdr)
        hdu3 = fits.ImageHDU(ks.velcirc, name="Velcirc", header=hdr)
        hdu4 = fits.ImageHDU(ks.velkin, name="VelKin", header=hdr)
        hdu5 = fits.ImageHDU(s_velo - ks.velcirc, name="V - VelKin", header=hdr)
        hdr["BUNIT"] = None

        out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
        out.writeto("MAGPI_Plots/plots/MAGPI" + field_name + "/fits_files/" + str(galaxy) + "_stellar_kinemetry.fits",
                    overwrite=True)
