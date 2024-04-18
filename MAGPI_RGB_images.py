import numpy as np
import pandas as pd
from astropy.visualization import make_lupton_rgb
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import os,shutil


def make_rgb_images(sample=None):
    file = pd.read_csv("MAGPI_csv/MAGPI_master_source_catalogue.csv", skiprows=16)
    if os.path.exists("kinemetry_sample_RGB"):
        shutil.rmtree("kinemetry_sample_RGB")
    os.mkdir("kinemetry_sample_RGB")
    if not sample is None:
        sample = pd.read_csv(sample)
        file = file[file["MAGPIID"].isin(sample['MAGPIID'])]
    else:
        pass
    z = file["z"].to_numpy()
    galaxy = file["MAGPIID"].to_numpy()
    for i in range(len(galaxy)):
        name = "MAGPI" + str(galaxy[i])
        try:
            file = fits.open("MAGPI_mini-images/"+name+"_mini-images.fits")
        except FileNotFoundError:
            print("No mini image!")
            continue
        DL = cosmo.luminosity_distance(z[i]).to(u.kpc).value
        pix = (np.degrees(10/DL)*3600)/0.2
        print("Beginning "+name+"...")
        g = file["GDATA"].data
        r = file['RDATA'].data
        i = file['IDATA'].data
        rgb = make_lupton_rgb(i, r, g)

        fig, ax = plt.subplots()
        ax.imshow(rgb, origin="lower")
        a = ax.get_xticks()
        b = ax.get_yticks()
        a = DL * np.radians((a * 0.2) / 3600)
        b = DL * np.radians((b * 0.2) / 3600)
        a = a - np.median(a)
        b = b - np.median(b)
        aa = [int(ii) for ii in a]
        bb = [int(ii) for ii in b]
        ax.set_xticklabels(aa)
        ax.set_yticklabels(bb)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.plot([2,2+pix],[2,2],color="w")
        ax.text(x=2.5,y=3,s=f"{10:.0f} kpc",color="w")
        ax.set_title(name)
        if not sample is None:
            plt.savefig("kinemetry_sample_RGB/" + name + ".pdf",bbox_inches="tight")
        else:
            plt.savefig("RGB_Images/" + name + ".pdf", bbox_inches="tight")
        print("Finishing " + name + "...")
    print("All Done!")



#make_rgb_images(sample="MAGPI_csv/MAGPI_kinemetry_sample_s05_bars_only.csv")
#make_rgb_images(sample="MAGPI_csv/MAGPI_kinemetry_sample_s05.csv")
make_rgb_images(sample="MAGPI_csv/MAGPI_kinemetry_sample_gas_dom.csv")
# make_rgb_images()
