#
# build
#

set(TEX_BASE_NAME bcs_nnrg_110308)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.pdf
    COMMAND
        ${CMAKE_SOURCE_DIR}/src/bash/run_xelatex.sh
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${TEX_BASE_NAME}.tex
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/${TEX_BASE_NAME}.tex
        #${CMAKE_CURRENT_SOURCE_DIR}/references.bib
        ${CMAKE_CURRENT_SOURCE_DIR}/parts/*.tex
        #${CMAKE_CURRENT_SOURCE_DIR}/figures/*
        #${CMAKE_CURRENT_SOURCE_DIR}/tables/*
    )

#add_custom_command(
    #OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.bbl
    #COMMAND
        #${CMAKE_SOURCE_DIR}/src/bash/run_bibtex.sh
        #${BIBTEX_COMPILER}
        #${CMAKE_CURRENT_SOURCE_DIR}
        #${TEX_BASE_NAME}
    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS
        #${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.aux
    #)

#add_custom_command(
    #OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.pdf
    #COMMAND
        #${CMAKE_SOURCE_DIR}/src/bash/run_pdflatex.sh
        #${PDFLATEX_COMPILER}
        #${CMAKE_CURRENT_SOURCE_DIR}
        #${CMAKE_CURRENT_SOURCE_DIR}/${TEX_BASE_NAME}.tex
    #COMMAND
        #${CMAKE_SOURCE_DIR}/src/bash/run_pdflatex.sh
        #${PDFLATEX_COMPILER}
        #${CMAKE_CURRENT_SOURCE_DIR}
        #${CMAKE_CURRENT_SOURCE_DIR}/${TEX_BASE_NAME}.tex
    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS
        #${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.bbl
    #)

add_custom_target(bcs_nnrg_110308 ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.pdf)

#
# install
#

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${TEX_BASE_NAME}.pdf
    DESTINATION .
    )

