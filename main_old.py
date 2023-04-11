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


def aperture_photometry(map, pa, a, b):
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
            if side1 + side2 < 8:
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
                    continue
                if np.log10(OIII[i, j] / HB[i, j]) > 1.30 + 0.72 / (
                        np.log10(SII[i, j] / HA[i, j]) - 0.32) and np.log10(
                    OIII[i, j] / HB[i, j]) < 1.89 * np.log10(SII[i, j] / HA[i, j]) + 0.76:
                    BPT_map[i, j] = 3
                    continue
                if np.log10(OIII[i, j] / HB[i, j]) > 1.30 + 0.72 / (
                        np.log10(SII[i, j] / HA[i, j]) - 0.32) and np.log10(
                    OIII[i, j] / HB[i, j]) > 1.89 * np.log10(SII[i, j] / HA[i, j]) + 0.76:
                    BPT_map[i, j] = 2
                    continue
                # if np.log10(OIII[i,j] / HB[i,j]) < 1.30 + 0.72 / (
                #         np.log10(SII[i,j] / HA[i,j]) - 0.32):
                #     BPT_map[i] = 1
                else:
                    pass
            # Outside 2Re
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


def BPT_plots(output_file, sample_file):
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
    galaxies = sample["MAGPIID"]
    for g in galaxies:
        print("Beginning MAGPI" + str(g) + "...\n")
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

        HA = clean_images(flux_Ha, pa, r50, r50 * q, img_err=flux_Ha / flux_Ha_err)
        HA_err = clean_images(flux_Ha_err, pa, r50, r50 * q)
        HB = clean_images(flux_Hb, pa, r50, r50 * q, img_err=flux_Hb / flux_Hb_err)
        OI = clean_images(OI, pa, r50, r50 * q, img_err=OI / OI_err)
        OIII = clean_images(OIII, pa, r50, r50 * q, img_err=OIII / OIII_err)
        NII = clean_images(NII, pa, r50, r50 * q, img_err=NII / NII_err)
        SII = clean_images(SII, pa, r50, r50 * q, img_err=SII / SII_err)

        if os.path.exists("plots/MAGPI" + str(g)[:4] + "/BPT_plots"):
            shutil.rmtree("plots/MAGPI" + str(g)[:4] + "/BPT_plots")
        os.mkdir("plots/MAGPI" + str(g)[:4] + "/BPT_plots")

        bpt_map = BPT_pixels(HA, NII, OI, OIII, HB, SII, pa, r50, r50 * q,
                             "plots/MAGPI" + str(g)[:4] + "/BPT_plots/" + str(g))

        fig, ax = plt.subplots()
        p = ax.imshow(bpt_map)
        cbar = plt.colorbar(p, ax=ax, ticks=[1, 2, 3])
        cbar.ax.set_yticklabels(["HII", "Seyfert", "LINER"])
        plt.savefig("plots/MAGPI" + str(g)[:4] + "/BPT_plots/" + str(g) + "bpt_map.pdf")

        HA_flux = aperture_photometry(HA, pa, 2 * r50, 2 * r50 * q)
        HA_err_flux = aperture_photometry(HA_err, pa, 2 * r50, 2 * r50 * q)
        HB_flux = aperture_photometry(HB, pa, 2 * r50, 2 * r50 * q)
        OIII_flux = aperture_photometry(OIII, pa, 2 * r50, 2 * r50 * q)
        NII_flux = aperture_photometry(NII, pa, 2 * r50, 2 * r50 * q)
        OI_flux = aperture_photometry(OI, pa, 2 * r50, 2 * r50 * q)
        SII_flux = aperture_photometry(SII, pa, 2 * r50, 2 * r50 * q)

        DL = cosmo.luminosity_distance(z).to(u.cm).value
        balmer = HA_flux / HB_flux
        HA_flux_corr = dust_corr(HA_flux * 1e-20, balmer, 6562 * u.AA, CCM89)
        HA_err_flux = dust_corr(HA_err_flux * 1e-20, balmer, 6562 * u.AA, CCM89)
        lum = HA_flux_corr * (4 * np.pi * DL ** 2)
        lum_err = HA_err_flux * (4 * np.pi * DL ** 2)
        SFR.append(lum * (10 ** (-41.27)))
        SFR_err.append(lum_err * (10 ** (-41.27)))

        HA_fluxes.append(HA_flux)
        HA_err_fluxes.append(HA_err_flux)
        HB_fluxes.append(HB_flux)
        OIII_fluxes.append(OIII_flux)
        NII_fluxes.append(NII_flux)
        OI_fluxes.append(OI_flux)
        SII_fluxes.append(SII_flux)
        re.append(r50)
        re_DL.append(np.radians(r50 / 3600) * DL * u.cm.to(u.kpc))

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
            # print(MAGPI[i], "LINER!")
            # print(bpt[i], "Match!")
            # count = count + 1
            sf_sy_ln[i] = 3
            continue
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.19 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.47)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and \
                np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.33 + (
                0.73 / (np.log10(OI_fluxes[i] / HA_fluxes[i]) + 0.59)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) < 1.89 * np.log10(SII_fluxes[i] / HA_fluxes[i]) + 0.76:
            # print(MAGPI[i], "LINER!")
            # print(bpt[i] == 3, "Match!")
            sf_sy_ln[i] = 3
            continue
        if np.log10(OI_fluxes[i] / HA_fluxes[i]) > -0.59 and np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.89 * np.log10(
                SII_fluxes[i] / HA_fluxes[i]) + 0.76 and np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.18 * np.log10(
            OI_fluxes[i] / HA_fluxes[i]) + 1.30:
            # print("Seyfert!")
            # print(bpt[i], "Match!")
            # count = count + 1
            sf_sy_ln[i] = 2
            continue
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.19 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.47)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and \
                np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.33 + (
                0.73 / (np.log10(OI_fluxes[i] / HA_fluxes[i]) + 0.59)):
            # print(MAGPI[i], "Seyfert!")
            # print(bpt[i], "Match!")
            # count = count + 1
            sf_sy_ln[i] = 2
            continue
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.05)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + (0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32)) and \
                np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.33 + (
                0.73 / (np.log10(OI_fluxes[i] / HA_fluxes[i]) - +0.59)):
            # print(MAGPI[i], "Star Forming!")
            # print(bpt[i], "Match!")
            # count = count + 1
            sf_sy_ln[i] = 1
            continue
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + (
                0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.05)) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.19 + 0.61 / (np.log10(NII_fluxes[i] / HA_fluxes[i]) - 0.47):
            # print(MAGPI[i], "Comp!")
            # print(bpt[i], "Match!")
            # count = count + 1
            sf_sy_ln[i] = 0
            continue
        else:
            # print(MAGPI[i], "Ambigious!")
            # print(bpt[i], "No Match!")
            sf_sy_ln[i] = 0
    SII_bpt = np.zeros(len(HA_fluxes))
    for i in range(len(HA_fluxes)):
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (
                np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) < 1.89 * np.log10(SII_fluxes[i] / HA_fluxes[i]) + 0.76:
            SII_bpt[i] = 3
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) > 1.30 + 0.72 / (
                np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32) and np.log10(
            OIII_fluxes[i] / HB_fluxes[i]) > 1.89 * np.log10(SII_fluxes[i] / HA_fluxes[i]) + 0.76:
            SII_bpt[i] = 2
        if np.log10(OIII_fluxes[i] / HB_fluxes[i]) < 1.30 + 0.72 / (np.log10(SII_fluxes[i] / HA_fluxes[i]) - 0.32):
            SII_bpt[i] = 1
    print("All Done!\n")

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
                       "SFR, dust corrected": SFR,
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
    plt.savefig("plots/BPT.pdf", bbox_inches="tight")

    return HA_fluxes, NII_fluxes, OIII_fluxes, HB_fluxes, SII_fluxes, OI_fluxes


def ellipse_plots(velo, velo_err, q, r50, field_name, file_name, output_file):
    os.mkdir(output_file + "/ellipse_plots")
    os.mkdir(output_file + "/ellipse_plots/ellipse_circ")
    y0, x0 = velo.shape
    y0, x0 = y0 / 2, x0 / 2

    start = (0.65 / 2) / 0.2
    step = (0.65 / 2) / 0.2
    end = 2 * r50
    rad = np.arange(start, end, step)

    k = kinemetry(img=velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                  bmodel=True, rangePA=[-120, 120], rangeQ=[q - 0.1, q + 0.1], allterms=True, cover=0.95)
    zeros = np.where(k.eccano == 0)[0]
    p = 0
    chi2 = []
    for i in range(2, len(zeros)):
        p += 1
        # k
        k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
        x = np.degrees(k.eccano[zeros[i - 1]:zeros[i]])
        ex_mom = k.ex_mom[zeros[i - 1]:zeros[i]]
        vrec = k.vrec[zeros[i - 1]:zeros[i]]
        vv = k.vv[zeros[i - 1]:zeros[i]]
        yEl = k.Yellip[zeros[i - 1]:zeros[i]].astype(int)
        xEl = k.Xellip[zeros[i - 1]:zeros[i]].astype(int)
        chi_2 = np.sum((ex_mom - vrec) ** 2 / (np.std(ex_mom) ** 2))
        chi_2 = chi2 / (len(ex_mom) - 2)
        chi2.append(chi_2)

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0})
        ax1.plot(x, vrec, c="firebrick", label="k," + r"$\chi_\nu^2$" + f":{chi2:.1e}")
        ax1.scatter(x, ex_mom, s=2)
        ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), c="k", zorder=2, s=3)
        ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=velo_err[yEl, xEl] / np.nanmax(k1), fmt=".k",
                     ls="",
                     zorder=1)
        ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), c="firebrick", zorder=1)
        ax1.scatter(x, ex_mom, s=2)
        ax1.plot(x, vrec, c="limegreen", label="M2")
        ax1.plot(x, vv, c="orangered", label=r"B$_1\cos\theta$")
        ax2.scatter(x, (ex_mom - vrec) / np.nanmax(k1), c="k", zorder=2, s=3)
        ax2.errorbar(x, (ex_mom - vrec) / np.nanmax(k1), yerr=velo_err[yEl, xEl] / np.nanmax(k1),
                     fmt=".k", ls="", zorder=1)
        ax2.plot(x, (ex_mom - vrec) / np.nanmax(k1), c="limegreen", zorder=1)
        ax2.plot(x, (ex_mom - vv) / np.nanmax(k1), c="orangered", zorder=1)
        ax2.scatter(x, (ex_mom - vv) / np.nanmax(k1), c="k", zorder=2, s=3)
        ax2.errorbar(x, (ex_mom - vv) / np.nanmax(k1), yerr=velo_err[yEl, xEl] / np.nanmax(k1),
                     fmt=".k", ls="", zorder=1)
        ax2.set_xlabel(r"$\theta$")
        ax2.set_ylabel(r"$\Delta{V}$")
        ax1.legend()
        ax1.set_ylabel(r"V [kms$^{-1}$]")
        # ax2.set_ylim(-0.75, 0.75)
        ax1.set_title(f'{file_name}, R={k.rad[i - 1] * 0.2:.2f}"({k.rad[i - 1] / r50[f]:.2f} Re)')
        ax2.hlines(y=0, xmin=x[0], xmax=x[-1], colors="gray", ls="dashdot")
        plt.savefig(output_file + "/ellipse_plots/ellipse_circ/ellipse_" + str(p) + ".pdf", bbox_inches="tight")


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


def stellar_gas_plots(field_name, cutoff, res_cutoff, n_ells, SNR_star, SNR_gas, mc=False, n=100, n_re=2):
    v_asym_ss = []
    v_asym_ss_err = []
    v_asym_gs = []
    v_asym_gs_err = []
    pa_gs = []
    pa_ss = []
    galaxies = []
    d_pas = []
    v_rot_s = []
    v_rot_g = []
    if os.path.exists("plots/" + field_name):
        shutil.rmtree("plots/" + field_name)
    os.mkdir("plots/" + field_name)
    os.mkdir("plots/" + field_name + "/flux_plots")
    os.mkdir("plots/" + field_name + "/fits_files")
    log_file = open("plots/" + field_name + "/" + field_name + "_log_file.txt", "w")
    csv_file = pd.read_csv("MAGPI_Emission_Lines/" + field_name + "/" + field_name + "_source_catalogue.csv",
                           skiprows=16)
    z = csv_file["z"].to_numpy()
    r50 = csv_file["R50_it"].to_numpy() / 0.2
    q = csv_file["axrat_it"].to_numpy()
    pa = csv_file["ang_it"].to_numpy()
    quality = csv_file["QOP"].to_numpy()
    file_name = csv_file["MAGPIID"].to_numpy()
    DL = cosmo.luminosity_distance(z).to(u.kpc).value
    pix = np.radians(0.33 / 3600) * DL
    for f in range(len(csv_file)):
        if z[f] > 0.35:
            print(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift not in range!")
            log_file.write(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
            continue
        elif z[f] < 0.28:
            print(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift not in range!")
            log_file.write(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift not in range!\n")
            continue
        elif quality[f] < 3:
            print(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift failed QC check!")
            log_file.write(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift failed QC check!\n")
            continue
        elif r50[f] < cutoff * res_cutoff:
            print(f"MAGPIID = {file_name[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!")
            log_file.write(f"MAGPIID = {file_name[f]}, r50 = {r50[f]:.2f} pix, not resolved enough!\n")
            continue
        # elif file_name[f] == int("1530197196") or file_name[f] == int("1502293058"):
        #     print(f"MAGPIID = {file_name[f]}, garbage galaxy\n")
        #     log_file.write(f"MAGPIID = {file_name[f]}, garbage galaxy\n")
        #     continue
        elif file_name[f] == int("1207128248") or file_name[f] == int("1506117050"):
            print(f"MAGPIID = {file_name[f]}, fixing PA\n")
            log_file.write(f"MAGPIID = {file_name[f]}, fixing PA\n")
            pa[f] = pa[f] - 90
        elif file_name[f] == int("1207197197"):
            print(f"MAGPIID = {file_name[f]}, fixing PA\n")
            log_file.write(f"MAGPIID = {file_name[f]}, fixing PA\n")
            pa[f] = pa[f] - 180
        elif file_name[f] == int("1204192193"):
            print(f"MAGPIID = {file_name[f]}, For Qainhui\n")
            log_file.write(f"MAGPIID = {file_name[f]}, For Qainhui\n")
        else:
            print(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift passed!")
            print(f"MAGPIID = {file_name[f]}, r50 = {r50[f]:.3f}, Res. passed!")
            print(f"MAGPIID = {file_name[f]} is {(r50[f] / res_cutoff):.3f} beam elements!")
            log_file.write(f"MAGPIID = {file_name[f]}, z = {z[f]:.3f}, Redshift passed!\n")
            log_file.write(f"MAGPIID = {file_name[f]}, r50 = {r50[f]:.3f}, Res. passed!\n")
            log_file.write(f"MAGPIID = {file_name[f]} is {(r50[f] / res_cutoff):.3f} beam elements!\n")

        star_file = "MAGPI_Absorption_Lines/" + field_name + "/galaxies/" + str(
            file_name[f]) + "_kinematics_ppxf-maps.fits"
        gas_file = "MAGPI_Emission_Lines/" + field_name + "/" + field_name + "_v2.2.1_GIST_EmissionLine_Maps/MAGPI" + str(
            file_name[f]) + "_GIST_EmissionLines.fits"

        if os.path.exists(gas_file) and os.path.exists(star_file):
            print("Has gas and stellar kinematics!")
            log_file.write("Has gas and stellar kinematics!\n")
            starfile = fits.open(star_file)
            gasfile = fits.open(gas_file)
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, \
                                                              gasfile[10].data, gasfile[11].data
            g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_velo_err = clean_images(g_velo_err, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            clip = np.nanmax(s_flux)
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Stellar SNR = {clip:.2f}...\n")
            log_file.write(f"Max Stellar SNR = {clip:.2f}...\n")
            if clip < SNR_star:
                print("Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                log_file.write(
                    "Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                log_file.write(
                    "Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                continue
            ha_check = np.count_nonzero(~np.isnan(g_flux))
            if ha_check < 50:
                print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " Ha spaxels survive!\n")
                log_file.write("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " Ha spaxels survive!\n")
                # print("Doing Kinemetry on stars only!")
                # log_file.write("Doing kinemetry on stars only!\n")
                # print("Cleaning up the gas map!\n")
                # log_file.write("Cleaning up the gas map!\n")
                # print("Doing Kinemetry on stars only")
                # log_file.write("Doing kinemetry on stars only!\n")
                print("Finding Brightest Line")
                log_file.write("Finding Brightest Line\n")
                max_line = pd.read_csv("MAGPI_csv/MAGPI_Emission_Max_Line.csv")
                max_line = max_line[max_line["MAGPIID"].isin([file_name[f]])]
                bright_line = max_line["MAX_LINE"].to_numpy()[0]
                print("Brightest line is " + bright_line)
                log_file.write("Brightest line is " + bright_line + "\n")
                bright_line_err = max_line["MAX_LINE"].to_numpy()[0]

                g_velo = gasfile[9].data
                g_flux = gasfile[bright_line].data
                g_flux_err = gasfile[bright_line_err].data
                g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
                g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)

                print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " spaxels survive!\n")
                log_file.write("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " spaxels survive!\n")
                bl_check = np.count_nonzero(~np.isnan(g_flux))
                if ha_check < bl_check:
                    print(bright_line + " is better!")
                    log_file.write(bright_line + " is better\n")
                else:
                    print(f"Ha is better")
                    log_file.write("Ha is better\n")
                    g_velo = gasfile[9].data
                    g_flux = gasfile[49].data
                    g_flux_err = gasfile[50].data
                    g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
                    g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)

                start = (0.65 / 2) / 0.2
                step = (0.65 / 2) / 0.2
                end = n_re * r50[f]
                rad = np.arange(start, end, step)
                if len(rad) < n_ells:
                    print("Not enough ellipses!")
                    log_file.write("Not enough ellipses!\n")
                    continue
                s_velo[np.isnan(s_velo)] = 0
                g_velo[np.isnan(g_velo)] = 0
                g_velo_err[np.isnan(g_velo_err)] = 0
                g_flux[np.isnan(g_flux)] = 0
                s_flux[np.isnan(s_flux)] = 0

                print("Doing kinemetry on stars and gas!")

                ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                               cover=0.95)
                kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                               cover=0.95)
                k_flux_g = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                     cover=0.95)
                k_flux_s = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                     cover=0.95)
                k_flux_s_k0 = k_flux_s.cf[:, 0]
                k_flux_g_k0 = k_flux_g.cf[:, 0]

                ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                ks3 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                ks4 = np.sqrt(ks.cf[:, 7] ** 2 + ks.cf[:, 8] ** 2)
                ks5 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
                v_asym_s = (ks2 + ks3 + ks4 + ks5) / (4 * ks1)

                kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                kg3 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                kg4 = np.sqrt(kg.cf[:, 7] ** 2 + kg.cf[:, 8] ** 2)
                kg5 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
                v_asym_g = (kg2 + kg3 + kg4 + kg5) / (4 * kg1)
                try:
                    v_asym_g_2re = np.nansum(k_flux_g_k0 * v_asym_g) / np.nansum(k_flux_g_k0)
                except ValueError:
                    continue
                try:
                    v_asym_s_2re = np.nansum(k_flux_s_k0 * v_asym_s) / np.nansum(k_flux_s_k0)
                except ValueError:
                    continue

                # pa_g = np.nansum(k_flux_g_k0 * kg.pa) / np.nansum(k_flux_g_k0)
                # pa_s = np.nansum(k_flux_s_k0 * ks.pa) / np.nansum(k_flux_s_k0)
                pa_g = np.nanmedian(kg.pa)
                pa_s = np.nanmedian(ks.pa)
                d_PA = np.abs(pa_g - pa_s)

                print(f"MAGPI{file_name[f]:.0f}")
                log_file.write(f"MAGPI{file_name[f]:.0f}\n")
                print(f"Gas Pa={pa_g:.1f}")
                log_file.write(f"Gas Pa={pa_g:.1f}\n")
                print(f"Stars Pa={pa_s:.1f}")
                log_file.write(f"Stars Pa={pa_s:.1f}\n")
                print(f"DeltaPA={d_PA:.2f}")
                log_file.write(f"DeltaPA={d_PA:.2f}\n")
                print(f"Stars={v_asym_s_2re:.2f}")
                log_file.write(f"Stars={v_asym_s_2re:.2f}\n")
                print(f"Gas={v_asym_g_2re:.2f}")
                log_file.write(f"Gas={v_asym_g_2re:.2f}\n")

                if mc:
                    v_asym_s_mc = np.zeros(n)
                    v_asym_g_mc = np.zeros(n)
                    for h in range(n):
                        print(f"{h + 1} Monte Carlo iteration...\n")
                        model_s = ks.velkin
                        model_g = kg.velkin
                        for i in range(len(s_velo[:, 0])):
                            for j in range(len(s_velo[0, :])):
                                model_s[i, j] = + np.random.normal(loc=s_velo[i, j], scale=s_velo_err[i, j])
                                model_g[i, j] = + np.random.normal(loc=g_velo[i, j], scale=g_velo_err[i, j])
                        kg = kinemetry(img=model_g, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                      bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                      fixcen=True)
                        ks = kinemetry(img=model_s, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                       bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                       fixcen=True)

                        try:
                            v_asym_s_mc[h] = np.nansum(k_flux_s_k0 * ((ks2 + ks3 + ks4 + ks5) / (4 * ks1))) / np.nansum(
                                k_flux_s_k0)
                        except ValueError:
                            v_asym_s_mc[h] = np.nan
                        try:
                            v_asym_g_mc[h] = np.nansum(k_flux_g_k0 * ((kg2 + kg3 + kg4 + kg5) / (4 * kg1))) / np.nansum(
                                k_flux_g_k0)
                        except ValueError:
                            v_asym_s_mc[h] = np.nan

                    galaxies.append(file_name[f])
                    v_asym_gs.append(np.nanmean(v_asym_g_mc))
                    v_asym_gs_err.append(np.nanstd(v_asym_g_mc))
                    v_asym_ss.append(np.nanmean(v_asym_s_mc))
                    v_asym_ss_err.append(np.nanstd(v_asym_s_mc))
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_PA)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nanmax(ks1))

                    fig, ax = plt.subplots()
                    ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                    ax.plot(ks.rad, ks1, zorder=1)
                    ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
                    ax.plot(kg.rad, kg1, zorder=1)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                                bbox_inches="tight")
                else:
                    galaxies.append(file_name[f])
                    v_asym_gs.append(v_asym_g_2re)
                    v_asym_ss.append(v_asym_s_2re)
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_PA)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nanmax(ks1))

                    fig, ax = plt.subplots()
                    ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                    ax.plot(ks.rad, ks1, zorder=1)
                    ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
                    ax.plot(kg.rad, kg1, zorder=1)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                                bbox_inches="tight")

                starfile.close()
                gasfile.close()
                starfile = fits.open(star_file)
                gasfile = fits.open(gas_file)

                if ha_check > bl_check:
                    s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data
                    g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[10].data, gasfile[11].data
                    g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f])
                    g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f])
                    g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f])
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
                    p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(s_sigma))
                    ax1.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    ax4.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                          height=2 * r50[f] / q[f], angle=pa_s, fc="none", ec="magenta"))
                    ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                          height=2 * r50[f] / q[f], angle=pa_g, fc="none", ec="magenta"))
                    ax1.set_ylabel("Stars")
                    ax4.set_ylabel("Gas")
                    for p, ax, label in zip([p1, p2, p3, p4, p5, p6], [ax1, ax2, ax3, ax4, ax5, ax6],
                                            [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]", r"SNR [H$\alpha$]",
                                             r"V [kms$^{-1}$]",
                                             r"$\sigma$ [kms$^{-1}$]"]):
                        plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                    # plt.savefig(output_file + "/" + str(galaxies) + "_fluxplots.pdf", bbox_inches="tight")
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_fluxplots.pdf",
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
                    hdr["OBJECT"] = str(file_name[f])
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
                    out.writeto("plots/" + field_name + "/fits_file/" + str(file_name[f]) + "_stellar_kinemetry.fits",
                                overwrite=True)
                else:
                    s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, \
                                                          starfile[4].data
                    g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[bright_line].data, gasfile[
                        bright_line_err].data, gasfile[
                                                                          9].data, gasfile[10].data, gasfile[11].data
                    g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f])
                    g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f])
                    g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f])
                    # g_flux = g_flux / g_flux_err
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
                    ax1.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    ax4.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                          height=2 * r50[f] / q[f], angle=pa_s, fc="none", ec="magenta"))
                    ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                          height=2 * r50[f] / q[f], angle=pa_g, fc="none", ec="magenta"))
                    ax1.set_ylabel("Stars")
                    ax4.set_ylabel("Gas")
                    for p, ax, label in zip([p1, p2, p3, p4, p5, p6], [ax1, ax2, ax3, ax4, ax5, ax6],
                                            [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]",
                                             bright_line[:-2] + " [x10$^{-20}$ erg s$^{-1}$ cm$^{-2}$]",
                                             r"V [kms$^{-1}$]",
                                             r"$\sigma$ [kms$^{-1}$]"]):
                        plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                    # plt.savefig(output_file + "/" + str(galaxies) + "_fluxplots.pdf", bbox_inches="tight")
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_fluxplots.pdf",
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
                    hdr["OBJECT"] = str(file_name[f])
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
                    out.writeto("plots/" + field_name + "/fits_files/" + str(file_name[f]) + "_stellar_kinemetry.fits",
                                overwrite=True)


            else:
                print("Doing kinemetry on stars and gas!")
                start = (0.65 / 2) / 0.2
                step = (0.65 / 2) / 0.2
                end = n_re * r50[f]
                rad = np.arange(start, end, step)
                if len(rad) < n_ells:
                    print("Not enough ellipses!")
                    log_file.write("Not enough ellipses!\n")
                    continue
                s_velo[np.isnan(s_velo)] = 0
                g_velo[np.isnan(g_velo)] = 0
                g_flux[np.isnan(g_flux)] = 0
                s_flux[np.isnan(s_flux)] = 0
                ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                               cover=0.95)
                kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                               cover=0.95)
                k_flux_g = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                     cover=0.95)
                k_flux_s = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                     cover=0.95)
                k_flux_s_k0 = k_flux_s.cf[:, 0]
                k_flux_g_k0 = k_flux_g.cf[:, 0]

                ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                ks3 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                ks4 = np.sqrt(ks.cf[:, 7] ** 2 + ks.cf[:, 8] ** 2)
                ks5 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
                v_asym_s = (ks2 + ks3 + ks4 + ks5) / (4 * ks1)

                kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                kg3 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                kg4 = np.sqrt(kg.cf[:, 7] ** 2 + kg.cf[:, 8] ** 2)
                kg5 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
                v_asym_g = (kg2 + kg3 + kg4 + kg5) / (4 * kg1)

                v_asym_g_2re = np.nansum(k_flux_g_k0 * v_asym_g) / np.nansum(k_flux_g_k0)
                v_asym_s_2re = np.nansum(k_flux_s_k0 * v_asym_s) / np.nansum(k_flux_s_k0)

                # pa_g = np.nansum(k_flux_g_k0 * kg.pa) / np.nansum(k_flux_g_k0)
                # pa_s = np.nansum(k_flux_s_k0 * ks.pa) / np.nansum(k_flux_s_k0)
                pa_g = np.nanmedian(kg.pa)
                pa_s = np.nanmedian(ks.pa)
                d_PA = np.abs(pa_g - pa_s)

                print(f"MAGPI{file_name[f]:.0f}")
                log_file.write(f"MAGPI{file_name[f]:.0f}\n")
                print(f"Gas Pa={pa_g:.1f}")
                log_file.write(f"Gas Pa={pa_g:.1f}\n")
                print(f"Stars Pa={pa_s:.1f}")
                log_file.write(f"Stars Pa={pa_s:.1f}\n")
                print(f"DeltaPA={d_PA:.2f}")
                log_file.write(f"DeltaPA={d_PA:.2f}\n")
                print(f"Stars={v_asym_s_2re:.2f}")
                log_file.write(f"Stars={v_asym_s_2re:.2f}\n")
                print(f"Gas={v_asym_g_2re:.2f}")
                log_file.write(f"Gas={v_asym_g_2re:.2f}\n")

                if mc:
                    v_asym_s_mc = np.zeros(n)
                    v_asym_g_mc = np.zeros(n)
                    for h in range(n):
                        print(f"{h + 1} Monte Carlo iteration...\n")
                        model_s = ks.velkin
                        model_g = kg.velkin
                        for i in range(len(s_velo[:, 0])):
                            for j in range(len(s_velo[0, :])):
                                model_s[i, j] = + np.random.normal(loc=s_velo[i, j], scale=s_velo_err[i, j])
                                model_g[i, j] = + np.random.normal(loc=g_velo[i, j], scale=g_velo_err[i, j])
                        ks = kinemetry(img=model_s, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                      bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                      fixcen=True)
                        kg = kinemetry(img=model_g, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                      bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                      fixcen=True)

                        # Compute Kinemetry Outputs
                        ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
                        ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
                        ks3 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
                        ks4 = np.sqrt(ks.cf[:, 7] ** 2 + ks.cf[:, 8] ** 2)
                        ks5 = np.sqrt(ks.cf[:, 9] ** 2 + ks.cf[:, 10] ** 2)

                        # Compute Kinemetry Outputs
                        kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                        kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                        kg3 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
                        kg4 = np.sqrt(kg.cf[:, 7] ** 2 + kg.cf[:, 8] ** 2)
                        kg5 = np.sqrt(kg.cf[:, 9] ** 2 + kg.cf[:, 10] ** 2)
                        try:
                            v_asym_s_mc[h] = np.nansum(k_flux_s_k0 * ((ks2 + ks3 + ks4 + ks5) / (4 * ks1))) / np.nansum(
                                k_flux_s_k0)
                        except ValueError:
                            v_asym_s_mc[h] = np.nan
                        try:
                            v_asym_g_mc[h] = np.nansum(k_flux_g_k0 * ((kg2 + kg3 + kg4 + kg5) / (4 * kg1))) / np.nansum(
                                k_flux_g_k0)
                        except ValueError:
                            v_asym_s_mc[h] = np.nan

                    galaxies.append(file_name[f])
                    v_asym_gs.append(np.nanmean(v_asym_g_mc))
                    v_asym_gs_err.append(np.nanstd(v_asym_g_mc))
                    v_asym_ss.append(np.nanmean(v_asym_s_mc))
                    v_asym_ss_err.append(np.nanstd(v_asym_s_mc))
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_PA)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nanmax(ks1))

                    fig, ax = plt.subplots()
                    ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                    ax.plot(ks.rad, ks1, zorder=1)
                    ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
                    ax.plot(kg.rad, kg1, zorder=1)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                                bbox_inches="tight")
                else:
                    galaxies.append(file_name[f])
                    v_asym_gs.append(v_asym_g_2re)
                    v_asym_ss.append(v_asym_s_2re)
                    pa_gs.append(pa_g)
                    pa_ss.append(pa_s)
                    d_pas.append(d_PA)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nanmax(ks1))

                    fig, ax = plt.subplots()
                    ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                    ax.plot(ks.rad, ks1, zorder=1)
                    ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
                    ax.plot(kg.rad, kg1, zorder=1)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                                bbox_inches="tight")

                starfile.close()
                gasfile.close()
                starfile = fits.open(star_file)
                gasfile = fits.open(gas_file)

                s_flux, s_velo, s_sigma = starfile[7].data, starfile[1].data, starfile[4].data
                g_flux, g_flux_err, g_velo, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data, gasfile[
                    11].data
                g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
                g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
                g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
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
                ax1.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                ax4.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                      height=2 * r50[f] / q[f], angle=pa_s, fc="none", ec="magenta"))
                ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                      height=2 * r50[f] / q[f], angle=pa_g, fc="none", ec="magenta"))
                ax1.set_ylabel("Stars")
                ax4.set_ylabel("Gas")
                for p, ax, label in zip([p1, p2, p3, p4, p5, p6], [ax1, ax2, ax3, ax4, ax5, ax6],
                                        [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]", r"SNR [H$\alpha$]",
                                         r"V [kms$^{-1}$]",
                                         r"$\sigma$ [kms$^{-1}$]"]):
                    plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                # plt.savefig(output_file + "/" + str(galaxies) + "_fluxplots.pdf", bbox_inches="tight")
                plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_fluxplots.pdf",
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
                hdr["OBJECT"] = str(file_name[f])
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
                out.writeto("plots/" + field_name + "/fits_files/" + str(file_name[f]) + "_stellar_kinemetry.fits",
                            overwrite=True)


        elif os.path.exists(gas_file) and os.path.exists(star_file) == False:
            print("Has gas kinematics but no stars!")
            log_file.write("Has gas kinematics but no stars!\n")
            gasfile = fits.open(gas_file)
            g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[9].data,gasfile[10].data,gasfile[11].data
            g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_velo_err = clean_images(g_velo_err, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
            g_flux = g_flux / g_flux_err

            clip = np.nanmax(g_flux)
            y0, x0 = g_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Gas SNR = {clip:.2f}...\n")
            log_file.write(f"Max Gas SNR = {clip:.2f}...\n")
            if clip < SNR_gas:
                print("Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                log_file.write(
                    "Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                log_file.write(
                    "Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                continue
            ha_check = np.count_nonzero(~np.isnan(g_flux))
            if ha_check < 50:
                print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " Ha spaxels survive!\n")
                log_file.write("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " Ha spaxels survive!\n")
                print("Finding Brightest Line")
                log_file.write("Finding Brightest Line\n")
                max_line = pd.read_csv("MAGPI_csv/MAGPI_Emission_Max_Line.csv")
                max_line = max_line[max_line["MAGPIID"].isin([file_name[f]])]
                bright_line = max_line["MAX_LINE"].to_numpy()[0]
                print("Brightest line is " + bright_line)
                log_file.write("Brightest line is " + bright_line + "\n")
                bright_line_err = max_line["MAX_LINE"].to_numpy()[0]

                g_velo = gasfile[9].data
                g_flux = gasfile[bright_line].data
                g_flux_err = gasfile[bright_line_err].data
                g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
                g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)

                print("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " spaxels survive!\n")
                log_file.write("Only " + str(np.count_nonzero(~np.isnan(g_flux))) + " spaxels survive!\n")
                bl_check = np.count_nonzero(~np.isnan(g_flux))
                if ha_check < bl_check:
                    print(bright_line + " is better!")
                    log_file.write(bright_line + " is better\n")
                else:
                    print(f"Ha is better")
                    log_file.write("Ha is better\n")
                    g_velo = gasfile[9].data
                    g_flux = gasfile[49].data
                    g_flux_err = gasfile[50].data
                    g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)
                    g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f], img_err=g_flux / g_flux_err)

                start = (0.65 / 2) / 0.2
                step = (0.65 / 2) / 0.2
                end = n_re * r50[f]
                rad = np.arange(start, end, step)
                if len(rad) < n_ells:
                    print("Not enough ellipses!")
                    log_file.write("Not enough ellipses!\n")
                    continue
                g_velo[np.isnan(g_velo)] = 0
                g_velo_err[np.isnan(g_velo_err)] = 0
                g_flux[np.isnan(g_flux)] = 0
                kg = kinemetry(img=g_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                               bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                               cover=0.95)
                k_flux_g = kinemetry(img=g_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                     bmodel=True,
                                     rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                     cover=0.95)
                k_flux_g_k0 = k_flux_g.cf[:, 0]

                kg1 = np.sqrt(kg.cf[:, 1] ** 2 + kg.cf[:, 2] ** 2)
                kg2 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                kg3 = np.sqrt(kg.cf[:, 3] ** 2 + kg.cf[:, 4] ** 2)
                kg4 = np.sqrt(kg.cf[:, 7] ** 2 + kg.cf[:, 8] ** 2)
                kg5 = np.sqrt(kg.cf[:, 5] ** 2 + kg.cf[:, 6] ** 2)
                v_asym_g = (kg2 + kg3 + kg4 + kg5) / (4 * kg1)
                try:
                    v_asym_g_2re = np.nansum(k_flux_g_k0 * v_asym_g) / np.nansum(k_flux_g_k0)
                except ValueError:
                    continue

                pa_g = np.nanmedian(kg.pa)

                print(f"MAGPI{file_name[f]:.0f}")
                log_file.write(f"MAGPI{file_name[f]:.0f}\n")
                print(f"Gas Pa={pa_g:.1f}")
                log_file.write(f"Gas Pa={pa_g:.1f}\n")
                print(f"Gas={v_asym_g_2re:.2f}")
                log_file.write(f"Gas={v_asym_g_2re:.2f}\n")

                if mc:
                    v_asym_g_mc = np.zeros(n)
                    for h in range(n):
                        print(f"{h + 1} Monte Carlo iteration...\n")
                        model_g = kg.velkin
                        for i in range(len(g_velo[:, 0])):
                            for j in range(len(g_velo[0, :])):
                                model_g[i, j] = + np.random.normal(loc=g_velo[i, j], scale=g_velo_err[i, j])
                        k = kinemetry(img=model_g, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                      bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                      fixcen=True)

                        # Compute Kinemetry Outputs
                        k1 = np.sqrt(k.cf[:, 1] ** 2 + k.cf[:, 2] ** 2)
                        k2 = np.sqrt(k.cf[:, 3] ** 2 + k.cf[:, 4] ** 2)
                        k3 = np.sqrt(k.cf[:, 5] ** 2 + k.cf[:, 6] ** 2)
                        k4 = np.sqrt(k.cf[:, 7] ** 2 + k.cf[:, 8] ** 2)
                        k5 = np.sqrt(k.cf[:, 9] ** 2 + k.cf[:, 10] ** 2)
                        try:
                            v_asym_g_mc[h] = np.nansum(k_flux_g_k0 * ((k2 + k3 + k4 + k5) / (4 * k1))) / np.nansum(
                                k_flux_g_k0)
                        except ValueError:
                            v_asym_g_mc[h] = np.nan

                    galaxies.append(file_name[f])
                    v_asym_ss.append(np.nan)
                    v_asym_ss_err.append(np.nan)
                    v_asym_gs.append(np.nanmean(v_asym_g_mc))
                    v_asym_gs_err.append(np.nanstd(v_asym_g_mc))
                    pa_ss.append(np.nan)
                    pa_gs.append(pa_g)
                    d_pas.append(np.nan)
                    v_rot_s.append(np.nan)
                    v_rot_g.append(np.nanmax(kg1))

                    fig, ax = plt.subplots()
                    # ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                    # ax.plot(ks.rad, ks1, zorder=1)
                    ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
                    ax.plot(kg.rad, kg1, zorder=1)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                                bbox_inches="tight")
                else:
                    galaxies.append(file_name[f])
                    v_asym_gs.append(v_asym_g_2re)
                    v_asym_ss.append(np.nan)
                    pa_gs.append(pa_g)
                    pa_ss.append(np.nan)
                    d_pas.append(np.nan)
                    v_rot_g.append(np.nanmax(kg1))
                    v_rot_s.append(np.nan)

                    fig, ax = plt.subplots()
                    # ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                    # ax.plot(ks.rad, ks1, zorder=1)
                    ax.scatter(kg.rad, kg1, ec="k", zorder=2, label="Gas")
                    ax.plot(kg.rad, kg1, zorder=1)
                    ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                    ax.set_xlabel("R [pix]")
                    ax.legend()
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                                bbox_inches="tight")

                    # pa_ss.append(pa_s)
                    # d_pas.append(d_PA)

                # starfile.close()
                gasfile.close()
                # starfile = fits.open(star_file)
                gasfile = fits.open(gas_file)
                if ha_check > bl_check:
                    # s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, \
                    #                                       starfile[4].data
                    g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[49].data, gasfile[50].data, gasfile[
                        9].data, gasfile[10].data, gasfile[11].data
                    g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f])
                    g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f])
                    g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f])
                    g_flux = g_flux / g_flux_err
                    # starfile.close()
                    gasfile.close()

                    fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(5, 8))
                    # p1 = ax1.imshow(s_flux, origin="lower")
                    # p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-0.5 * np.nanmax(s_velo),
                    #                 vmax=0.5 * np.nanmax(s_velo))
                    # p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
                    p4 = ax4.imshow(g_flux, origin="lower")
                    p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                                    vmax=0.9 * np.nanmax(g_velo))
                    p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(s_sigma))
                    # ax1.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    ax4.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    # ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                    #                       height=2 * r50[f] / q[f], angle=pa_s, fc="none", ec="magenta"))
                    ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                          height=2 * r50[f] / q[f], angle=pa_g, fc="none", ec="magenta"))
                    # ax1.set_ylabel("Stars")
                    ax4.set_ylabel("Gas")
                    for p, ax, label in zip([p4, p5, p6], [ax4, ax5, ax6],
                                            [r"SNR [H$_\alpha$]",
                                             r"V [kms$^{-1}$]",
                                             r"$\sigma$ [kms$^{-1}$]"]):
                        plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                    # plt.savefig(output_file + "/" + str(galaxies) + "_fluxplots.pdf", bbox_inches="tight")
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_fluxplots.pdf",
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
                    hdr["OBJECT"] = file_name[f]
                    n = None
                    hdu0 = fits.PrimaryHDU(n, header=hdr)
                    hdu1 = fits.ImageHDU(g_flux, name="SNR_Stars", header=hdr)
                    hdu2 = fits.ImageHDU(g_velo, name="Data", header=hdr)
                    hdu3 = fits.ImageHDU(kg.velcirc, name="Velcirc", header=hdr)
                    hdu4 = fits.ImageHDU(kg.velkin, name="VelKin", header=hdr)
                    hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
                    hdr["BUNIT"] = None

                    out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
                    out.writeto("plots/" + field_name + "/fits_files/" + str(file_name[f]) + "_stellar_kinemetry.fits",
                                overwrite=True)

                else:
                    # s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, \
                    #                                       starfile[4].data
                    g_flux, g_flux_err, g_velo, g_velo_err, g_sigma = gasfile[bright_line].data, gasfile[
                        bright_line_err].data, gasfile[
                                                                          9].data, gasfile[10].data, gasfile[11].data
                    g_velo = clean_images(g_velo, pa[f], r50[f], r50[f] * q[f])
                    g_sigma = clean_images(g_sigma, pa[f], r50[f], r50[f] * q[f])
                    g_flux = clean_images(g_flux, pa[f], r50[f], r50[f] * q[f])
                    # g_flux = g_flux / g_flux_err
                    # starfile.close()
                    gasfile.close()

                    fig, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(5, 8))
                    # p1 = ax1.imshow(s_flux, origin="lower")
                    # p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-0.5 * np.nanmax(s_velo),
                    #                 vmax=0.5 * np.nanmax(s_velo))
                    # p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
                    p4 = ax4.imshow(g_flux, origin="lower")
                    p5 = ax5.imshow(g_velo, origin="lower", cmap="cmr.redshift", vmin=-0.9 * np.nanmax(g_velo),
                                    vmax=0.9 * np.nanmax(g_velo))
                    p6 = ax6.imshow(g_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.2 * np.nanmax(s_sigma))
                    # ax1.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    ax4.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
                    # ax2.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                    #                       height=2 * r50[f] / q[f], angle=pa_s, fc="none", ec="magenta"))
                    ax5.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                          height=2 * r50[f] / q[f], angle=pa_g, fc="none", ec="magenta"))
                    # ax1.set_ylabel("Stars")
                    ax4.set_ylabel("Gas")
                    for p, ax, label in zip([p4, p5, p6], [ax4, ax5, ax6],
                                            [bright_line[:-2] + r" [x10$^{-20}$ erg s$^{-1}$ cm$^{-2}$]",
                                             r"V [kms$^{-1}$]",
                                             r"$\sigma$ [kms$^{-1}$]"]):
                        plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
                    # plt.savefig(output_file + "/" + str(galaxies) + "_fluxplots.pdf", bbox_inches="tight")
                    plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_fluxplots.pdf",
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
                    hdr["OBJECT"] = file_name[f]
                    n = None
                    hdu0 = fits.PrimaryHDU(n, header=hdr)
                    hdu1 = fits.ImageHDU(g_flux, name="SNR_Stars", header=hdr)
                    hdu2 = fits.ImageHDU(g_velo, name="Data", header=hdr)
                    hdu3 = fits.ImageHDU(kg.velcirc, name="Velcirc", header=hdr)
                    hdu4 = fits.ImageHDU(kg.velkin, name="VelKin", header=hdr)
                    hdu5 = fits.ImageHDU(g_velo - kg.velcirc, name="V - VelKin", header=hdr)
                    hdr["BUNIT"] = None

                    out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
                    out.writeto("plots/" + field_name + "/fits_files/" + str(file_name[f]) + "_stellar_kinemetry.fits",
                                overwrite=True)


        elif os.path.exists(gas_file) == False and os.path.exists(star_file):
            print("Has stellar kinematics but no gas!")
            log_file.write("Has stellar kinematics but no gas!\n")
            starfile = fits.open(star_file)
            s_flux, s_velo, s_velo_err, s_sigma = starfile[7].data, starfile[1].data, starfile[3].data, starfile[4].data

            clip = np.nanmax(s_flux)
            y0, x0 = s_flux.shape
            x0 = int(x0 / 2)
            y0 = int(y0 / 2)
            print(f"Max Stellar SNR = {clip:.2f}...\n")
            log_file.write(f"Max Stellar SNR = {clip:.2f}...\n")
            if clip < SNR_star:
                print("Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                log_file.write(
                    "Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                continue
            elif np.isinf(clip) or np.isnan(clip):
                print("Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                log_file.write(
                    "Not Plotting or doing Kinemetry on " + str(file_name[f]) + " because its heinous looking\n")
                continue

            start = (0.65 / 2) / 0.2
            step = (0.65 / 2) / 0.2
            end = n_re * r50[f]
            rad = np.arange(start, end, step)
            if len(rad) < n_ells:
                print("Not enough ellipses!")
                log_file.write("Not enough ellipses!\n")
                continue
            s_velo[np.isnan(s_velo)] = 0
            s_flux[np.isnan(s_flux)] = 0
            ks = kinemetry(img=s_velo, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                           bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                           cover=0.95)
            k_flux_s = kinemetry(img=s_flux, x0=x0, y0=y0, ntrm=10, plot=False, verbose=False, radius=rad,
                                 bmodel=True,
                                 rangePA=[pa[f] - 10, pa[f] + 10], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                 cover=0.95)
            k_flux_s_k0 = k_flux_s.cf[:, 0]
            ks1 = np.sqrt(ks.cf[:, 1] ** 2 + ks.cf[:, 2] ** 2)
            ks2 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks3 = np.sqrt(ks.cf[:, 3] ** 2 + ks.cf[:, 4] ** 2)
            ks4 = np.sqrt(ks.cf[:, 7] ** 2 + ks.cf[:, 8] ** 2)
            ks5 = np.sqrt(ks.cf[:, 5] ** 2 + ks.cf[:, 6] ** 2)
            v_asym_s = (ks2 + ks3 + ks4 + ks5) / (4 * ks1)
            try:
                v_asym_s_2re = np.nansum(k_flux_s_k0 * v_asym_s) / np.nansum(k_flux_s_k0)
            except ValueError:
                continue

            # pa_s = np.nansum(k_flux_s_k0 * ks.pa) / np.nansum(k_flux_s_k0)
            pa_s = np.nanmedian(ks.pa)
            print(f"MAGPI{file_name[f]:.0f}")
            log_file.write(f"MAGPI{file_name[f]:.0f}\n")
            print(f"Stars Pa={pa_s:.1f}")
            log_file.write(f"Stars Pa={pa_s:.1f}\n")
            print(f"Stars={v_asym_s_2re:.2f}")
            log_file.write(f"Stars={v_asym_s_2re:.2f}\n")

            if mc:
                v_asym_s_mc = np.zeros(n)
                for h in range(n):
                    print(f"{h + 1} Monte Carlo iteration...\n")
                    model_s = ks.velkin
                    for i in range(len(s_velo[:, 0])):
                        for j in range(len(s_velo[0, :])):
                            model_s[i, j] = + np.random.normal(loc=s_velo[i, j], scale=s_velo_err[i, j])
                    ks_mc = kinemetry(img=model_s, x0=x0, y0=y0, ntrm=11, plot=False, verbose=False, radius=rad,
                                      bmodel=True, rangePA=[-120, 120], rangeQ=[q[f] - 0.1, q[f] + 0.1], allterms=True,
                                      fixcen=True)

                    # Compute Kinemetry Outputs
                    ks1 = np.sqrt(ks_mc.cf[:, 1] ** 2 + ks_mc.cf[:, 2] ** 2)
                    ks2 = np.sqrt(ks_mc.cf[:, 3] ** 2 + ks_mc.cf[:, 4] ** 2)
                    ks3 = np.sqrt(ks_mc.cf[:, 3] ** 2 + ks_mc.cf[:, 4] ** 2)
                    ks4 = np.sqrt(ks_mc.cf[:, 7] ** 2 + ks_mc.cf[:, 8] ** 2)
                    ks5 = np.sqrt(ks_mc.cf[:, 5] ** 2 + ks_mc.cf[:, 6] ** 2)
                    v_asym_s = (ks2 + ks3 + ks4 + ks5) / (4 * ks1)

                    v_asym_s_mc[h] = np.nansum(k_flux_s_k0 * v_asym_s) / np.nansum(k_flux_s_k0)

                galaxies.append(file_name[f])
                v_asym_gs.append(np.nan)
                v_asym_gs_err.append(np.nan)
                v_asym_ss.append(np.nanmean(v_asym_s_mc))
                v_asym_ss_err.append(np.nanstd(v_asym_s_mc))
                pa_gs.append(np.nan)
                pa_ss.append(pa_s)
                d_pas.append(np.nan)
                v_rot_g.append(np.nan)
                v_rot_s.append(np.nanmax(ks1))

                fig, ax = plt.subplots()
                ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                ax.plot(ks.rad, ks1, zorder=1)
                ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                ax.set_xlabel("R [pix]")
                ax.legend()
                plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                            bbox_inches="tight")
            else:
                galaxies.append(file_name[f])
                v_asym_gs.append(np.nan)
                v_asym_ss.append(v_asym_s_2re)
                pa_gs.append(np.nan)
                pa_ss.append(pa_s)
                d_pas.append(np.nan)
                v_rot_g.append(np.nan)
                v_rot_s.append(np.nanmax(ks1))

                fig, ax = plt.subplots()
                ax.scatter(ks.rad, ks1, ec="k", zorder=2, label="Stars")
                ax.plot(ks.rad, ks1, zorder=1)
                ax.set_ylabel(r"V$_{rot}$ [kms$^{-1}$]")
                ax.set_xlabel("R [pix]")
                ax.legend()
                plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_Vrot.pdf",
                            bbox_inches="tight")

            starfile.close()
            starfile = fits.open(star_file)
            s_flux, s_velo, s_sigma = starfile[7].data, starfile[1].data, starfile[4].data
            starfile.close()

            fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10, 8))
            p1 = ax1.imshow(s_flux, origin="lower")
            p2 = ax2.imshow(s_velo, origin="lower", cmap="cmr.redshift", vmin=-220, vmax=220)
            p3 = ax3.imshow(s_sigma, origin="lower", cmap="copper", vmin=0, vmax=0.5 * np.nanmax(s_sigma))
            ax1.add_patch(Circle(xy=(pix[f], pix[f]), radius=pix[f], fc="none", ec="k"))
            ax1.add_patch(Ellipse(xy=(x0, y0), width=2 * r50[f],
                                  height=2 * r50[f] / q[f], angle=pa_s, fc="none", ec="magenta"))
            ax1.set_ylabel("Stars")
            for p, ax, label in zip([p1, p2, p3], [ax1, ax2, ax3],
                                    [r"SNR", r"V [kms$^{-1}$]", r"$\sigma$ [kms$^{-1}$]", r"SNR",
                                     r"V [kms$^{-1}$]",
                                     r"$\sigma$ [kms$^{-1}$]"]):
                plt.colorbar(p, ax=ax, label=label, pad=0, fraction=0.047, location="top")
            # plt.savefig(output_file + "/" + str(galaxies) + "_fluxplots.pdf", bbox_inches="tight")
            plt.savefig("plots/" + field_name + "/flux_plots/" + str(file_name[f]) + "_fluxplots.pdf",
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
            hdr["OBJECT"] = str(file_name[f])
            n = None
            hdu0 = fits.PrimaryHDU(n, header=hdr)
            hdu1 = fits.ImageHDU(s_flux, name="SNR_Stars", header=hdr)
            hdu2 = fits.ImageHDU(s_velo, name="Data", header=hdr)
            hdu3 = fits.ImageHDU(ks.velcirc, name="Velcirc", header=hdr)
            hdu4 = fits.ImageHDU(ks.velkin, name="VelKin", header=hdr)
            hdu5 = fits.ImageHDU(s_velo - ks.velcirc, name="V - VelKin", header=hdr)
            hdr["BUNIT"] = None

            out = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
            out.writeto("plots/" + field_name + "/fits_files/" + str(file_name[f]) + "_stellar_kinemetry.fits",
                        overwrite=True)
    if mc == False:
        v_asym_ss_err = list(np.zeros(len(galaxies)))
        v_asym_gs_err = list(np.zeros(len(galaxies)))

    return galaxies, v_asym_gs, v_asym_gs_err, v_asym_ss, v_asym_ss_err, pa_gs, pa_ss, d_pas, v_rot_g, v_rot_s


# galaxies = []
# v_asym_gs = []
# v_asym_ss = []
# v_asym_g_errs = []
# v_asym_s_errs = []
# pa_gs = []
# pa_ss = []
# d_pas = []
# vrot_ss = []
# vrot_gs = []
# fields = glob.glob("MAGPI_Emission_Lines/M*")
# mc =False
# for i in range(len(fields)):
#     magpi = fields[i].split("/")[-1]
#     res_cutoff = (0.65 / 2) / 0.2
#     galaxy, v_asym_g, v_asym_g_err, v_asym_s, v_asym_s_err, pa_g, pa_s, d_pa, vrot_g, vrot_s = stellar_gas_plots(magpi,
#                                                                                                                  cutoff=1,
#                                                                                                                  res_cutoff=res_cutoff,
#                                                                                                                  n_ells=5,
#                                                                                                                  SNR_gas=20,
#                                                                                                                  SNR_star=3,
#                                                                                                                  mc=mc,
#                                                                                                                  n=100,
#                                                                                                                  n_re=2)
#     galaxies.append(galaxy)
#     v_asym_ss.append(v_asym_s)
#     v_asym_s_errs.append(v_asym_s_err)
#     v_asym_g_errs.append(v_asym_g_err)
#     v_asym_gs.append(v_asym_g)
#     pa_gs.append(pa_g)
#     pa_ss.append(pa_s)
#     d_pas.append(d_pa)
#     vrot_ss.append(vrot_s)
#     vrot_gs.append(vrot_g)
#
# new_galaxies = []
# new_v_asym_gs = []
# new_v_asym_ss = []
# new_v_asym_g_errs = []
# new_v_asym_s_errs = []
# new_pa_gs = []
# new_pa_ss = []
# new_d_pas = []
# new_vrot_ss = []
# new_vrot_gs = []
# new_galaxies = list_flat(galaxies, new_galaxies)
# new_v_asym_gs = list_flat(v_asym_gs, new_v_asym_gs)
# new_v_asym_s_errs = list_flat(v_asym_s_errs, new_v_asym_s_errs)
# new_v_asym_g_errs = list_flat(v_asym_g_errs, new_v_asym_g_errs)
# new_v_asym_ss = list_flat(v_asym_ss, new_v_asym_ss)
# new_pa_ss = list_flat(pa_ss, new_pa_ss)
# new_pa_gs = list_flat(pa_gs, new_pa_gs)
# new_d_pas = list_flat(d_pas, new_d_pas)
# new_vrot_ss = list_flat(vrot_ss, new_vrot_ss)
# new_vrot_gs = list_flat(vrot_gs, new_vrot_gs)
#
# df = pd.DataFrame({"MAGPIID": new_galaxies,
#                    "v_asym_gas": new_v_asym_gs,
#                    "v_asym_gas_err": new_v_asym_g_errs,
#                    "v_asym_stars": new_v_asym_ss,
#                    "v_asym_stars_err": new_v_asym_s_errs,
#                    "PA_gas (1Re)": new_pa_gs,
#                    "PA_stars (1Re)": new_pa_ss,
#                    "DeltaPA (1Re)": new_d_pas,
#                    "Vrot_Gas": new_vrot_gs,
#                    "Vrot_Stars": new_vrot_ss})
# # df = df.dropna()
# # df = df[df["v_asym_gas"]/df["v_asym_gas_err"] > 3]
# print("Sample size is " + str(len(df)) + "!\n")
# if mc==True:
#     df.to_csv("MAGPI_csv/MAGPI_kinemetry_sample.csv", index=False)
# if mc==False:
#     df.to_csv("MAGPI_csv/MAPGI_kinemetry_sample_no_err.csv",index=False)
#
# BPT_plots("MAGPI_csv/MAGPI_kinemetry_sample_BPT.csv", "MAGPI_csv/MAGPI_kinemetry_sample.csv")
