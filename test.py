from math import cos, pi, sin
import cv2
import glob
import numpy as np


#random things I need to test before implementing in solution

pot = "Support Files/ear_data/test/"
absolutna_pot = "/home/anze/Documents/Faks/Slikovna biometrija/DrugaDomacaNaloga/sbNaloga2/Support Files/ear_data/test/"

poti_do_slik = glob.glob(pot + '*.png')
poti_do_resnic = glob.glob(pot + '*.txt')

print(poti_do_resnic)

