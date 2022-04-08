import cv2
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
import pydicom
import pydicom._storage_sopclass_uids
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
import math


def prepare_mask(length):
    mask = np.empty(length, dtype=float)
    val = (-4 / math.pow(np.pi, 2)) * 255
    for i in range(length):
        if i % 2 == 0:
            mask[i] = 0.0
        else:
            mask[i] = val / math.pow(i, 2)
    mask[0] = 255
    return mask

def convert_image_to_ubyte(img):
    return img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))


def readDicom(filename):
    dicom = pydicom.read_file(filename)
    dicomImg = np.array(dicom.pixel_array)
    height, width = dicomImg.shape
    return dicomImg,height,width,dicom

def read_file(filename):
    file= cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    height, width =file.shape
    return file,height,width

def prepare_image(file,height,width):
    if height!=width:
        size=max(height,width)
        temp_file=np.zeros((size,size),np.uint8)
        padding_x = int((size-width)/2)
        padding_y = int((size - height) / 2)
        temp_file[padding_y:padding_y+height,padding_x:padding_x+width]=file
        return temp_file
    return file

def Emiter_postion(img,angle):
    angle = np.deg2rad(angle)
    r = img.shape[0] * np.sqrt(2) / 2
    center = int(img.shape[0] / 2)
    emiter = [int(r * np.cos(angle)) + center, int(r * np.sin(angle)) + center]
    return emiter

def Detectors_postion(img,angle,detectors,detectorsAngle):
    angle = np.deg2rad(angle)
    positions = []
    r = img.shape[0] * np.sqrt(2) / 2
    center = int(img.shape[0] / 2)
    if detectors > 1:
        for i in range(detectors):
            position = [
                int(r * np.cos(angle + np.pi - detectorsAngle / 2 + i * detectorsAngle / (detectors - 1))) + center,
                int(r * np.sin(angle + np.pi - detectorsAngle / 2 + i * detectorsAngle / (detectors - 1))) + center]
            positions.append(position)
    return positions



def bresenham(x0, y0, x1, y1):
    line=[]
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0
    D = 2*dy - dx
    y = 0
    for x in range(dx + 1):
        line.append([x0 + x*xx + y*yx, y0 + x*xy + y*yy])
        if D >= 0:
            y = y+1
            D = D-2*dx
        D = D+2*dy
    return line

def SumofPixels(emiter,detectors,img):
    list_of_values=[]
    for detecor in detectors:
        suma = 0
        counter=0
        points=bresenham(emiter[0],emiter[1],detecor[0],detecor[1])
        for point in points:
            if ValidPoint(img,point[0],point[1])==True:
                suma=suma+img[point[0]][point[1]]
                counter=counter+1
        if counter!=0:
            list_of_values.append(np.float64(suma/counter))
        else:
            list_of_values.append(np.float64(0))
    return list_of_values


def Fill_Image(sinogram,emiter,detectors,image,image_helper):
    for idx,detector in enumerate(detectors):
        suma=0
        counter=0
        points=bresenham(emiter[0],emiter[1],detector[0],detector[1])
        for point in points:
            if ValidPoint(image_helper, point[0], point[1]) == True:
                suma=suma+image[point[0]][point[1]]
                counter=counter+1
        for point in points:
            if counter!=0:
                value=(sinogram[idx]-suma)/counter
                if ValidPoint(image_helper, point[0], point[1]) == True:
                    image[point[0]][point[1]] = image[point[0]][point[1]] + value
    return image

# Alternative algorithm to fill the image based on sinogram values
# def Fill_Image2(sinogram,emiter,detectors,image,image_helper):
#     for idx,detector in enumerate(detectors):
#         points=bresenham(emiter[0],emiter[1],detector[0],detector[1])
#         value=sinogram[idx]
#         for point in points:
#             if ValidPoint(image_helper, point[0], point[1]) == True:
#                 image[point[0]][point[1]] = image[point[0]][point[1]] + value
#     return image


def Sinogram(img,number_of_iteration,number_of_detectors,detectorAngle,filtered):
    detectorAngle = 2 * np.deg2rad(detectorAngle)
    sinogram=[]
    angles=np.linspace(0,360,number_of_iteration)
    for idx,ang in enumerate(angles):
        emiter=Emiter_postion(img,ang)
        detectors=Detectors_postion(img,ang,number_of_detectors,detectorAngle)
        values=SumofPixels(emiter,detectors,img)
        if filtered == True:
            maska=prepare_mask(len(values))
            values = np.convolve(values, maska)
        sinogram.append(values)
    sinogram=np.array(sinogram).transpose()
    return sinogram

def Sinogram_iter(img,number_of_iteration,number_of_detectors,detectorAngle,filtered,stop):
    detectorAngle = 2 * np.deg2rad(detectorAngle)
    sinogram=[]
    angles=np.linspace(0,360,number_of_iteration)
    for idx,ang in enumerate(angles):
        if idx==stop*10:
            sinogram = np.array(sinogram).transpose()
            return sinogram
        emiter=Emiter_postion(img,ang)
        detectors=Detectors_postion(img,ang,number_of_detectors,detectorAngle)
        values=SumofPixels(emiter,detectors,img)
        if filtered == True:
            values = np.convolve(values, [-2, 5, -2])
        sinogram.append(values)
    sinogram=np.array(sinogram).transpose()
    return sinogram

def ValidPoint(img,x,y):
    if x>=img.shape[0] or y>=img.shape[0] or x<0 or y<0:
        return False
    return True


def Reconstruction_Technique(image,sinogram,number_of_iteration,number_of_detectors,detectorAngle):
    detectorAngle = 2 * np.deg2rad(detectorAngle)
    img=[[0 for i in range(image.shape[0])] for j in range(image.shape[0])]
    angles = np.linspace(0, 360, number_of_iteration)
    for idx,ang in enumerate(angles):
        emiter = Emiter_postion(image, ang)
        detectors = Detectors_postion(image, ang, number_of_detectors, detectorAngle)
        values = sinogram[:,idx]
        img=Fill_Image(values,emiter,detectors,img,image)
    return img

def mse_error(image_in,image_out):
    mse_error_arr=[]
    glob_sum=0
    for i in range(image_in.shape[0]):
        row_sum=0
        for j in range(image_in.shape[0]):
            diff=(image_in[i][j]-image_out[i][j])**0.5
            row_sum=row_sum+diff
            glob_sum=glob_sum+diff
        mse_error_arr.append(row_sum/image_in.shape[0])
    glob_mse_error=glob_sum/(image_in.shape[0]*image_in.shape[0])
    return glob_mse_error,mse_error


if __name__=="__main__":
    st.title('Tomograph Simulator')
    file=st.file_uploader("Upload DICOM file below!")
    st.title("Tomograph parameters")
    number_of_iteration = st.slider("Number of iteration",max_value=200,value=120)
    number_of_detectors = st.slider("Number of detectors",max_value=200,value=120)
    angle = st.slider("Angle of rotation",max_value=360,value=90)
    filtered=st.checkbox("Filter")
    checkbox = st.checkbox('Shows steps of creating sinogram')
    if checkbox:
        img, h, w, dicom = readDicom(file)
        img = prepare_image(img, h, w)
        iters = int(number_of_iteration / 10)
        which_iteration = st.slider("Iteration step: ", max_value=iters,min_value=1)
        sinogram_iter = Sinogram_iter(img, number_of_iteration, number_of_detectors, angle, filtered, which_iteration)
        fig, ax = plt.subplots()
        fig.set_size_inches(2.5,2.5)
        ax.imshow(sinogram_iter, cmap=plt.cm.Greys_r)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        tittle="Sinogram after "+str(which_iteration*10)+" iteration"
        ax.set_title(tittle)
        st.pyplot(fig)

    with st.form("Patient data"):
        st.title("Patient data")
        left, right = st.columns(2)
        with left:
            name = st.text_input("Name")
        with right:
            lastname=st.text_input("Last name")
        left, right = st.columns(2)
        with left:
            date_of_examination=st.date_input("Date of examination")
        with right:
            sex=st.selectbox("Select gender",('Man','Woman'))
        comment = st.text_input("Additional Comment")
        submitted = st.form_submit_button("Submit")

        if submitted:
            img, h, w,dicom= readDicom(file)
            img = prepare_image(img, h, w)
            sinogram = Sinogram(img, number_of_iteration, number_of_detectors, angle, filtered)
            reconstructed_image = Reconstruction_Technique(img, sinogram, number_of_iteration, number_of_detectors,
                                                           angle)
            sinogram = (np.array(sinogram) / np.array(sinogram).sum()) / np.array(sinogram).max()
            reconstructed_image = (np.array(reconstructed_image) / np.array(reconstructed_image).sum()) / np.array(
                reconstructed_image).max()
            fig, ((ax1), (ax2),(ax4)) = plt.subplots(1, 3, figsize=(10, 10))
            ax1.imshow(img, cmap=plt.cm.Greys_r)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            ax1.set_title("Orginal image")
            ax2.imshow(sinogram, cmap=plt.cm.Greys_r)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax2.set_title("Sinogram")
            ax4.imshow(reconstructed_image, cmap=plt.cm.Greys_r)
            ax4.set_title("Recreated image")
            ax4.axes.get_xaxis().set_visible(False)
            ax4.axes.get_yaxis().set_visible(False)
            st.pyplot(fig)
            dicom.PatientName = str(name) + str(lastname)
            dicom.PatientSex = str(sex)
            dicom.TreatmentDate = str(date_of_examination)
            dicom.TextComments = str(comment)
            reconstructed_image=convert_image_to_ubyte(reconstructed_image)
            dicom.PixelData=reconstructed_image.tobytes()
            dicom.Rows=reconstructed_image.shape[0]
            dicom.Columns=reconstructed_image.shape[0]
            dicom.save_as("ZAPISANY.dcm")









