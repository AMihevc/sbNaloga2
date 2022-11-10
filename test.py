from math import cos, pi, sin
import cv2
import glob
import numpy as np


#random things I need to test before implementing in solution


def dodaj_resnico(slika, pot_do_slike):

    pot_do_resnice = pot_do_slike[:len(pot_do_slike)-3] + 'txt'

    with open (pot_do_resnice, 'rt') as resnica:
        for vrstica in resnica:
            _ , resnica_center_x , resnica_center_y ,resnica_širina , resnica_višina = [float(i) for i in vrstica.rstrip("\n").split(" ")]

    slika_h, slika_w, _ = slika.shape

    #pretvorba v pixel vrednosti širine in višine 
    resnica_širina = resnica_širina * slika_w
    resnica_višina = resnica_višina * slika_h

    #pretvorba v pixel vrednosti in postavitev na levi zgornji rob 
    resnica_center_x = resnica_center_x * slika_w - (resnica_širina/2)
    resnica_center_y = resnica_center_y * slika_h - (resnica_višina/2)

    #izris kvadrata 
    zacetek_kvadrata = (round(resnica_center_x), round(resnica_center_y))
    konec_kvadrata = (round(resnica_center_x + resnica_širina), round( resnica_center_y + resnica_višina))
    barva_kvadrata = (64,170,0)
    debelina_kvadrata = 5

    slika_s_kvadratom = cv2.rectangle(slika, zacetek_kvadrata, konec_kvadrata, barva_kvadrata, debelina_kvadrata)


    return slika_s_kvadratom

def izrisi_sliko(slika):

    cv2.imshow('IZRIS SLIKE', slika)

    #kle morš pol prtisnt neko tipko da se zapre
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0


#pravilna pot do slik in resnic 
pot = "Support Files/ear_data/test/"
absolutna_pot = "/home/anze/Documents/Faks/Slikovna biometrija/DrugaDomacaNaloga/sbNaloga2/Support Files/ear_data/test/"

poti_do_slik = glob.glob(pot + '*.png')

#preberi eno sliko

pot_do_slike = "Support Files/ear_data/test/0501.png"
slika = cv2.imread(pot_do_slike, 1)
#slika_rgb = cv2.cvtColor(slika,cv2.COLOR_BGR2RGB)

slika = dodaj_resnico(slika, pot_do_slike)

#izris slike


izrisi_sliko(slika)