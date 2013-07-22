///////////////////////////////////////////////////////////////////////////////
// glInfo.h
// ========
// get GL vendor, version, supported extensions and other states using glGet*
// functions and store them glInfo struct variable
//
// To get valid OpenGL infos, OpenGL rendering context (RC) must be opened
// before calling glInfo::getInfo(). Otherwise it returns false.
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2005-10-04
// UPDATED: 2009-10-07
//
// Copyright (c) 2005 Song Ho Ahn
//
// Modified by Ohannessian H. Gorune on 24/12/2010
// -Uses set<string> extensionSet instead of vector<string> extensions;
///////////////////////////////////////////////////////////////////////////////

#ifndef GLINFO_H
#define GLINFO_H

#include <string>
#include <set>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "common_vis.h"

using namespace std;

// struct to store OpenGL info
struct glInfo
{
    string vendor;
    string renderer;
    string version;
	set<string> extensionsSet;	// My change
    int redBits;
    int greenBits;
    int blueBits;
    int alphaBits;
    int depthBits;
    int stencilBits;
    int maxTextureSize;
    int maxLights;
    int maxAttribStacks;
    int maxModelViewStacks;
    int maxProjectionStacks;
    int maxClipPlanes;
    int maxTextureStacks;

    // ctor, init all members
    glInfo() : redBits(0), greenBits(0), blueBits(0), alphaBits(0), depthBits(0),
               stencilBits(0), maxTextureSize(0), maxLights(0), maxAttribStacks(0),
               maxModelViewStacks(0), maxClipPlanes(0), maxTextureStacks(0) {};

    bool getInfo();									// extract info
    void printSelf();								// print itself
    bool isExtensionSupported(const string& ext);	// check if a extension is supported
};
#endif