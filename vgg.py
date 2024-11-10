from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# 事前学習済みのVGG16をロード
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16に全結合層を追加
x = model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model_final = Model(inputs=model.input, outputs=predictions)

# コンパイル
model_final.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 学習
model_final.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))
