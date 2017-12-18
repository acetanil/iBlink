cmake_minimum_required(VERSION 2.8)
project(iBlink_X_SVM)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(pca ./src/make_lbp.cpp)
target_link_libraries(pca ${OpenCV_LIBS})


add_executable(train ./src/train_svm.cpp)
target_link_libraries(train ${OpenCV_LIBS})


add_executable(sampling ./src/sampling.cpp)
target_link_libraries(sampling ${OpenCV_LIBS})


# add_executable(predict ./src/predict.cpp)
# target_link_libraries(predict ${OpenCV_LIBS})

add_executable(predict_list ./src/predict_list.cpp)
target_link_libraries(predict_list ${OpenCV_LIBS})

add_executable(main ./src/predict_cam.cpp)
target_link_libraries(main ${OpenCV_LIBS})


add_executable(find_bounding_box ./src/find_bounding_box.cpp)
target_link_libraries(find_bounding_box ${OpenCV_LIBS})


# add_executable(draw ./src/draw.cpp)
# target_link_libraries(draw ${OpenCV_LIBS})
