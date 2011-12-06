cdef extern from "SgSystem.h":
    pass

cdef extern from "SgInit.h":
    void SgInit()

cdef extern from "SgPoint.h":
    ctypedef int SgGrid
    ctypedef int SgPoint

cdef extern from "SgPoint.h" namespace "SgPointUtil":
    SgPoint Pt(int column, int row)
    SgGrid Row(SgPoint point)
    SgGrid Col(SgPoint point)

cdef extern from "SgBlackWhite.h":
    ctypedef int SgBlackWhite

    enum BlackWhiteConstant:
        SG_BLACK
        SG_WHITE

cdef extern from "SgBoardColor.h":
    ctypedef int SgBoardColor

    enum BoardColorConstant:
        SG_EMPTY
        SG_BORDER

cdef extern from "GoInit.h":
    void GoInit()

cdef extern from "GoBoard.h":
    enum GoMoveInfoFlag:
        GO_MOVEFLAG_REPETITION
        GO_MOVEFLAG_SUICIDE
        GO_MOVEFLAG_CAPTURING
        GO_MOVEFLAG_ILLEGAL

    cdef cppclass GoBoard:
        GoBoard(int size)

        SgBoardColor GetColor(SgPoint point)
        SgBlackWhite ToPlay()
        bint IsLegal(int point)
        bint IsLegal(int point, SgBlackWhite player)
        void Play(SgPoint point)
        void Play(SgPoint point, SgBlackWhite player)
        bint LastMoveInfo(GoMoveInfoFlag flag)

