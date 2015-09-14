#ifdef _CREATEDLL_
#define DLL_API _declspec( dllexport)
#else
#define DLL_API _declspec( dllimport)
#endif

#include <vector>
#include "function.h"
#include "frame.h"
#include "Cblock.h"
#include <atlstr.h>

extern "C"
{
	void DLL_API run(CString m_mfcFolder, CString m_fileName, int level, int m_nFrame, float percent, CvSize quality );
};

