from math import cos, pi, sin
import cv2
import glob
import numpy as np


#Code for the solution of the assigment
'''TODO
implment yolo5
    - odpreš vse slike z glob
    - za vsako slikoo runaš model iz yolo 
    - primerjaš z ground truth, ki je v .txt filih za vsako sliko

implement Haar
    - odpreš vsako lsiko z glob
'''

#Definicije funkcij

#Funkcija najde vse slike in resnice in vrne seznam poti do slik in resnic
def slike(pot):
    poti_do_slik = glob.glob(pot + '*.png')
    poti_do_resnic = glob.glob(pot + '*.txt')

    return poti_do_slik, poti_do_resnic







# ------------------------------------------ Začetek "izvajanja" programa --------------------------------------------------------------------------
pot_do_testnih_podatkov = pot = "Support Files/ear_data/test/"