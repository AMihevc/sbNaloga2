from math import cos, pi, sin
import cv2
import glob
import numpy as np
import torch

#random things I need to test before implementing in solution

#naredi novo sliko z dodano resnico
def dodaj_resnico( pot_do_slike):
    slika_s_kvadratom = cv2.imread(pot_do_slike, 1)
    pot_do_resnice = pot_do_slike[:len(pot_do_slike)-3] + 'txt'

    with open (pot_do_resnice, 'rt') as resnica:
        for vrstica in resnica:
            _ , resnica_center_x , resnica_center_y ,resnica_širina , resnica_višina = [float(i) for i in vrstica.rstrip("\n").split(" ")]

    slika_h, slika_w, _ = slika_s_kvadratom.shape

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

    slika_s_kvadratom = cv2.rectangle(slika_s_kvadratom, zacetek_kvadrata, konec_kvadrata, barva_kvadrata, debelina_kvadrata)


    return slika_s_kvadratom


def dobi_resnico(pot_do_slike):
    slika = cv2.imread(pot_do_slike, 1)
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

    #vrni obliko (x1,y1,x2,y2)
    return [int(round(resnica_center_x)), int(round(resnica_center_y)), int(round(resnica_center_x + resnica_širina)), int(round(resnica_center_y + resnica_višina))]


#izriše podano sliko
def izrisi_sliko(slika):

    cv2.imshow('IZRIS SLIKE', slika)

    #kle morš pol prtisnt neko tipko da se zapre
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return 0

#funkcija na sliko doda skatle
#skatla mora biti oblike (x1,y1,x2,y2)
def dodaj_skatlo(slika, skatla, barva):
    if len(skatla) == 0:
        return 0
    for (x1,y1,x2,y2) in skatla:
        cv2.rectangle(slika, (x1, y1),(x2, y2), barva, 5)
    return 0

#funkcija izračuna VJ skatle s podanimi parametri in jih vrne v obliki [x,y,w,h], [x,y,w,h], ...
def izracunaj_Haar(pot_do_slike, scaleFaktor, minSosedov, levo_uho_filter_VJ, desno_uho_filter_VJ):

    #odpri sliko v 
    slika_grey = cv2.imread(pot_do_slike, 0)



    #zaznaj ušesa
    levo_uho_skatla = levo_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = scaleFaktor, minNeighbors = minSosedov)
    desno_uho_skatla = desno_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = scaleFaktor, minNeighbors = minSosedov)

    #če najdeš obe ušesi vrni oboje kot en seznam
    if(len(levo_uho_skatla) > 0 and len(desno_uho_skatla) > 0):
        return np.concatenate([levo_uho_skatla,desno_uho_skatla])
    
    #če ne najdeš levih vrni samo desna
    if (len(levo_uho_skatla) < 1):
        return desno_uho_skatla

    #če ne najdeš desnih vrni samo leva
    if (len(desno_uho_skatla) < 1):
        return levo_uho_skatla

    #če ne najdeš nič vrni prazen seznam 
    return [[]]

#funkcija izracuna iou med dvema skatlama
def get_iou(skatla1, skatla2):
    """
    izračunaj IOU med škatlama 

    Vhod:
    skatla1 : seznam['x1', 'y1', 'x2', 'y2']
        (x1, y1) lev zgornji kot skatle 
        (x2, y2) desn spodnji kot skatle 

    skatla2 : seznam['x1', 'y1', 'x2', 'y2']
        (x1, y1) lev zgornji kot skatle 
        (x2, y2) desn spodnji kot skatle 

    Vrne IOU (float)
    """
    #koordinate območja prekrivanja 
    x_levo = max(skatla1[0], skatla2[0])
    y_zgoraj = max(skatla1[1], skatla2[1])
    x_desno = min(skatla1[2], skatla2[2])
    y_spodaj = min(skatla1[3], skatla2[3])

    #če se ne prekrivata vrni 0 
    if x_desno < x_levo or y_spodaj < y_zgoraj:
        return 0.0

    # površina območja prekrivanja
    obmocje_prekrivanja = (x_desno - x_levo) * (y_spodaj - y_zgoraj)

    #površina obeh škatel
    skatla1_povrsina = (skatla1[2] - skatla1[0]) * (skatla1[3] - skatla1[1])
    skatla2_povrsina = (skatla2[2] - skatla2[0]) * (skatla2[3] - skatla2[1])

    #iou škatel
    iou = obmocje_prekrivanja / float(skatla1_povrsina + skatla2_povrsina - obmocje_prekrivanja)

    #iou more bit med 0 in 1
    assert iou >= 0.0
    assert iou <= 1.0

    return iou

#funkcija pretvori obliko skatel iz [x,y,w,h] v [x1,y1,x2,y2]
def pretvor_obliko(seznam_skatel):
    seznam_novih_oblik = []
    #če je seznam prazen vrni prazen seznam 
    if len(seznam_skatel) == 0:
        #print("zaznal sem da je prazen seznam")
        return []

    for (x, y ,w ,h )in seznam_skatel:
            nova_oblika = []
            nova_oblika.append(x)
            nova_oblika.append(y)
            nova_oblika.append(x + w)
            nova_oblika.append(y + h)
            seznam_novih_oblik.append(nova_oblika)

    return seznam_novih_oblik

#funkcija izracuna yolo skatle in jih vrne v obliki [x1,y1,x2,y2], [x1,y1,x2,y2], ...
def yolo_model_skatle(pot_do_slike, model):
    

    rezultat_yolo = model(pot_do_slike)
    skatle_yolo = []
    for skatla in rezultat_yolo.xyxy[0].tolist():
        x1, y1, x2, y2, sigurnost, _ = [float(i) for i in skatla]
        skatla_yolo = [round(x1), round(y1), round(x2), round(y2)]
        skatle_yolo.append(skatla_yolo)

    return skatle_yolo

#funkcija pridobi tp fp in fn za dolocen threshold. Vrne tuple (tp, fp, fn)
def dobi_tpfpfn(resnica, seznam_skatel, threshold): 
    tp = 0
    fp = 0
    fn = 0
    
    # če nisi nič zaznal je to False Negative fn++
    if len(seznam_skatel) == 0:
        fn += 1
    
    #če si nekaj zaznal
    else:
        for skatla in seznam_skatel:
            #izracunaj iou skatle
            iou = get_iou(resnica, skatla)

            #če je iou 0 (ni prekrivanja) je to FP
            if iou == 0:
                fp += 1
            #če je prekrivanje 
            else: 
                #in je znotraj thresholda je to TP
                if iou >= threshold:
                    #tu moraš pazit da je lahko samo en tp na sliko
                    if tp >= 1:
                        #ostale šteješ za FP 
                        fp += 1 
                    else:
                        tp += 1 
                #in ni znotraj thresholda je to FP
                elif iou < threshold:
                    fp += 1 

    return tp, fp, fn

#funkcija obdela vse slike 
def obdelaj_slike(poti_do_slik, scaleFactor, stSosedov, threshold):

    #naloži filtre 
    levo_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_leftear.xml')
    desno_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_rightear.xml')

    #model  = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")
    # za vse slike dub škatle in zračuni
    yolo_global_tpfpfn = (0, 0, 0)
    vj_global_tpfpfn = (0, 0, 0)
    counter_slik = 0
    for pot_do_slike in poti_do_slik:
        
        #pridobi resnico škatle
        resnica = dobi_resnico(pot_do_slike)

        #pridobi seznam škatel yolo. Oblika [[x1, y1, x2, y2],[skatla2], ...]
        #skatle_yolo = yolo_model_skatle(pot_do_slike,model)

        #pridobi seznam škatel VJ. Oblika [[x,y,w,h]]
        skatle_VJ = izracunaj_Haar(pot_do_slike, scaleFactor, stSosedov, levo_uho_filter_VJ, desno_uho_filter_VJ)

        #pretvori VJ skatle v obliko [[x1, y1, x2, y2],[skatla2], ...]]
        skatle_VJ = pretvor_obliko(skatle_VJ)

        #izrisi rezultate
        #resnica je ZELENA 
        #slika_z_resnico = dodaj_resnico(pot_do_slike)
        #dodam yolo skatle (MODRE)
        #dodaj_skatlo(slika_z_resnico,skatle_yolo,(255,0,0))
        #dodam VJ skatle (RDEČE)
        #dodaj_skatlo(slika_z_resnico,skatle_VJ,(0,0,255))
    	#izrisem sliko
        #izrisi_sliko(slika_z_resnico)

        #pridobi tp, fp, fn za yolo 
        #yolo_tpfpfn = dobi_tpfpfn(resnica,skatle_yolo,threshold)
        #print(yolo_tpfpfn)

        #pridobi tp, fp, fn za VJ
        vj_tpfpfn = dobi_tpfpfn(resnica,skatle_VJ,threshold)
        #print(vj_tpfpfn)

        #yolo_global_tpfpfn = tuple(map(sum, zip(yolo_tpfpfn, yolo_global_tpfpfn)))
        print(counter_slik)
        counter_slik += 1

        vj_global_tpfpfn = tuple(map(sum, zip(vj_tpfpfn, vj_global_tpfpfn)))

    return [yolo_global_tpfpfn, vj_global_tpfpfn]



#pravilna pot do slik in resnic 
pot = "Support Files/ear_data/test/"
absolutna_pot = "/home/anze/Documents/Faks/Slikovna biometrija/DrugaDomacaNaloga/sbNaloga2/Support Files/ear_data/test/"

poti_do_slik = glob.glob(pot + '*.png')

#preberi eno sliko
#pot_do_mrBean = poti_do_slik[0] # mr.bean
#slika = cv2.imread(pot_do_mrBean, 1)

#mr bean z resnico 
#slika_z_resnico = dodaj_resnico(pot_do_mrBean)

# yolov5 model ----------------------------------------
'''
model_yolo = torch.hub.load( 'yolov5', 'custom', path='Support Files/yolo5s.pt', source="local")

rezultat_yolo = model_yolo(pot_do_slike)

rezultat_yolo.show()

'''

#skatle_yolo = yolo_model_skatle(pot_do_mrBean)
#print("skatle_yolo")
#print(skatle_yolo)

#dodaj_skatlo(slika_z_resnico,skatle_yolo,(255,0,0))

#haar-cascade --------------------------------------------
'''
#sliko pretvori v gray scale 
slika_grey = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

#naloži modele za levo in desno uho 

levo_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_leftear.xml')
desno_uho_filter_VJ = cv2.CascadeClassifier('Support Files\haarcascade_mcs_rightear.xml')

levo_uho_skatla = levo_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = 1.2, minNeighbors = 5)
desno_uho_skatla = desno_uho_filter_VJ.detectMultiScale(slika_grey, scaleFactor = 1.1, minNeighbors = 5)

#print(levo_uho_skatla)
#print(desno_uho_skatla)

'''

#izračunam Haar skatle 
#zdruzen_seznam = izracunaj_Haar(pot_do_mrBean, 1.03, 4)
#print("Zdruzen seznam:")
#print(zdruzen_seznam)

#skatle_VJ = pretvor_obliko(zdruzen_seznam)
#print("skatle_VJ")
#print(skatle_VJ)

#samo za izris
#dodaj_skatlo(slika_z_resnico, skatle_VJ, (0,0,255))

#izrisi_sliko(slika_z_resnico)

#za računanje IOU 
#resnica = dobi_resnico(pot_do_mrBean)
#print(resnica)

#iou = get_iou(resnica,VJ_seznam[0])
#print(iou)

#pridobi tp, fp, fn za yolo 
#yolo_tpfpfn = dobi_tpfpfn(resnica, skatle_yolo, 0.5)

#pridobi tp, fp, fn za VJ
'''
vj_tpfpfn = dobi_tpfpfn(resnica, skatle_VJ, 0.5)
print("yolo_tpfpfn")
print(yolo_tpfpfn)
print("vj_tpfpfn")
print(vj_tpfpfn)
'''
#izrisi_sliko(slika_z_resnico)

#optimizacija pri 0.5 treshold za iou

deset_slik = [poti_do_slik[0],poti_do_slik[1],poti_do_slik[2], poti_do_slik[3], poti_do_slik[4],poti_do_slik[5],poti_do_slik[6],poti_do_slik[7],poti_do_slik[8],poti_do_slik[9],poti_do_slik[10]]

dve_sliki = [poti_do_slik[0],poti_do_slik[1]]

rezultati_tpfpfn = obdelaj_slike(poti_do_slik, 1.01, 0, 0.5)

print(rezultati_tpfpfn)

'''
yolo 
"0.5": [157,67,298]
vj 
"0.5": [157,67,298]

'''
TP = rezultati_tpfpfn[0][0]
FP = rezultati_tpfpfn[0][1]
FN = rezultati_tpfpfn[0][2]

yolo_iou = rezultati_tpfpfn[0][0] / (sum(rezultati_tpfpfn[0]))
vj_iou = rezultati_tpfpfn[1][0] / (sum(rezultati_tpfpfn[1]))

print(yolo_iou)
print(vj_iou)

precsion = TP / TP+FP
recall = TP / TP+FN

print("Precision" + str(precsion))

print("Recall" + str(recall))
