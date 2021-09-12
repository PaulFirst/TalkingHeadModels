import os
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import interface
import cv2
import matplotlib.pyplot as plt

import torch
import emb_inference
import FT
import inference


class MainWindow(QtWidgets.QDialog, interface.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.core.setText(self.getDevice())
        self.stackedWidget.setCurrentIndex(0)
        self.exitBtn.clicked.connect(self.close_window)
        self.imagePP.clicked.connect(self.setStep1Image)
        self.videoPP.clicked.connect(self.setStep1Video)
        self.step1BackBtn.clicked.connect(self.backToMenu)
        self.step2BackBtn.clicked.connect(self.backToMenu)
        self.step3BackBtn.clicked.connect(self.backToMenu)
        self.step2CheckResultBtn.clicked.connect(self.checkFTResult)

        self.image_path = ''
        self.video_path = ''
        self.path_to_image = ''
        self.path_to_video = ''

    def getDevice(self):
        return 'GPU(cuda)' if torch.cuda.is_available() else 'CPU'

    def close_window(self):
        self.close()

    def setStep1Image(self):
        self.stackedWidget.setCurrentIndex(1)
        self.step1TitleLabel.setText('Генерация изображения')
        self.step1InfoLabel.setText('Шаг 1. Выбор фотографии образа для обучения \n сети признакам внешности')
        self.step1ChosenLabel.setText('Выбранные фото:')
        self.step1NextBtn.clicked.connect(self.setStep2Image)
        self.step1AcceptBtn.clicked.connect(self.openDialogImages)
        self.step1ListWidget.clear()
        self.step1NextBtn.setEnabled(False)

    def openDialogImages(self):
        self.image_path = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Dialog', '', '*.jpg')[0]
        self.step1ListWidget.addItems(self.image_path)

        print(self.image_path)
        print(self.image_path[0])
        self.image_path = os.path.dirname(self.image_path[0])
        print(self.image_path)
        if self.step1ListWidget.count() != 0:
            self.step1NextBtn.setEnabled(True)

    def setStep1Video(self):
        self.stackedWidget.setCurrentIndex(1)
        self.step1TitleLabel.setText('Генерация видео')
        self.step1InfoLabel.setText('Шаг 1. Выбор видео образа для обучения \n сети признакам внешности')
        self.step1ChosenLabel.setText('Выбранное видео:')
        self.step1NextBtn.clicked.connect(self.setStep2Video)
        self.step1AcceptBtn.clicked.connect(self.openDialogVideo)
        self.step1ListWidget.clear()
        self.step1NextBtn.setEnabled(False)

    def openDialogVideo(self):
        self.video_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Dialog', '', '*.mp4')[0]
        print(self.video_path)
        self.step1ListWidget.addItem(self.video_path)
        if self.step1ListWidget.count() != 0:
            self.step1NextBtn.setEnabled(True)

    def setStep2Image(self):
        self.stackedWidget.setCurrentIndex(2)
        self.step2TitleLabel.setText('Генерация изображения')
        self.step2InfoLabel.setText('Шаг 2. Настройка сети для работы с текущими признаками внешности')
        self.step2NextBtn.clicked.connect(self.setStep3Image)
        self.step2ItersTextEdit.setText('10')
        self.step2FTLabel.setText('')
        self.step2FTBtn.clicked.connect(self.fineTuningImages)

    def fineTuningImages(self):
        embedder = emb_inference.EmbInference(path_to_images=self.image_path)
        self.step2FTLabel.setText('Загрузка изображений...')
        embedder.loadingImageInput()
        self.step2FTLabel.setText('Загрузка компонента сети...')
        embedder.loadingEmbedder()
        self.step2FTLabel.setText('Построение вектора...')
        embedder.makeEmbeddingImage()
        self.step2FTLabel.setText('Построение вектора завершено!')
        epochs = self.step2ItersTextEdit.toPlainText()
        ft = FT.FT(path_to_images=self.image_path, epochs=int(epochs), path_to_embedding='e_hat_images.tar')
        ft.createImageDataset()
        self.step2FTLabel.setText('Загрузка компонента сети...')
        ft.loadModel()
        self.step2FTLabel.setText('Настрока сети...')
        ft.fineTuning()
        self.step2FTLabel.setText('Настрока сети завершена. Параметры сохранены!')

    def checkFTResult(self):
        img = cv2.imread('fig.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.draw()
        plt.show()
        print(img)

    def setStep2Video(self):
        self.stackedWidget.setCurrentIndex(2)
        self.step2TitleLabel.setText('Генерация видео')
        self.step2InfoLabel.setText('Шаг 2. Настройка сети для работы с текущими признаками внешности')
        self.step2NextBtn.clicked.connect(self.setStep2Video)
        self.step2ItersTextEdit.setText('10')
        self.step2FTLabel.setText('')
        self.step2FTBtn.clicked.connect(self.fineTuningVideo)
        self.step2NextBtn.clicked.connect(self.setStep3Video)

    def fineTuningVideo(self):
        embedder = emb_inference.EmbInference(path_to_video=self.video_path)
        self.step2FTLabel.setText('Загрузка видео...')
        embedder.loadingVideoInput()
        self.step2FTLabel.setText('Загрузка компонента сети...')
        embedder.loadingEmbedder()
        self.step2FTLabel.setText('Построение вектора...')
        embedder.makeEmbeddingVideo()
        self.step2FTLabel.setText('Построение вектора завершено!')
        epochs = self.step2ItersTextEdit.toPlainText()
        ft = FT.FT(path_to_video=self.image_path, epochs=int(epochs), path_to_embedding='e_hat_video.tar')
        ft.createVideoDataset()
        self.step2FTLabel.setText('Загрузка компонента сети...')
        ft.loadModel()
        self.step2FTLabel.setText('Настрока сети...')
        ft.fineTuning()
        self.step2FTLabel.setText('Настрока сети завершена. Параметры сохранены!')

    def setStep3Image(self):
        self.stackedWidget.setCurrentIndex(3)
        self.step3PreprocessBtn.clicked.connect(self.generateImage)
        self.step3CheckResultBtn.clicked.connect(self.checkResultImage)
        self.step3ChooseBtn.clicked.connect(self.chooseDriveImage)
        self.step3PreprocessLabel.setText('')
        self.step3PreprocessLabel.setText('Генерация завершена. Изображение сохранено!')

    def generateImage(self):
        self.step3PreprocessLabel.setText('Выполнение...')
        infer = inference.Inference('e_hat_images.tar', self.path_to_image)
        infer.generateImage()
        self.step3PreprocessLabel.setText('Генерация завершена. Изображение сохранено!')
        self.step2TitleLabel.setText('Генерация изображения')
        self.step2InfoLabel.setText('Шаг 3. Выполнение генерации и получение итогового изображения')
        self.step3ChosenLabel.setText('Выбранный управляющий кадр')

    def chooseDriveImage(self):
        self.path_to_image = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Dialog', '', '*.jpg')[0]
        self.step3TextEdit.setText(self.path_to_image)

    def checkResultImage(self):
        img = cv2.imread('result_image.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.draw()
        plt.show()

    def setStep3Video(self):
        self.stackedWidget.setCurrentIndex(3)
        self.step3CheckResultBtn.clicked.connect(self.checkResultVideo)
        self.step3PreprocessLabel.setText('')
        self.step3PreprocessBtn.clicked.connect(self.generateVideo)
        self.step3ChooseBtn.clicked.connect(self.chooseDriveVideo)
        self.step3PreprocessLabel.setText('Генерация завершена. Видео сохранено!')
        self.step3TitleLabel.setText('Генерация видео')
        self.step3InfoLabel.setText('Шаг 3. Выполнение генерации и получение итогового видео')
        self.step3ChosenLabel.setText('Выбранная управляющая последовательность')

    def chooseDriveVideo(self):
        self.path_to_video = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Dialog', '', '*.mp4')[0]
        self.step3TextEdit.setText(self.path_to_video)

    def generateVideo(self):
        self.step3PreprocessLabel.setText('Выполнение...')
        infer = inference.Inference('e_hat_video.tar', self.path_to_video)
        infer.generateVideo()
        self.step3PreprocessLabel.setText('Генерация завершена. Видео сохранено!')

    def checkResultVideo(self):
        cap = cv2.VideoCapture('result_video.mp4')
        ret, frame = cap.read()
        while 1:
            ret, frame = cap.read()
            cv2.imshow('Результат', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or ret is False:
                cap.release()
                cv2.destroyAllWindows()
                break
            cv2.imshow('Результат', frame)

    def backToMenu(self):
        self.stackedWidget.setCurrentIndex(0)


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainWindow()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
