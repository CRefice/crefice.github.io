+++
title = "Projects | Carlo Refice"
description = "The projects I have been working on"

template = "projects.html"
+++

# Personal Projects

{% card(image="/projects/prayer.png", alt="A scene rendered with Prayer") %}

## [Prayer](https://github.com/CRefice/prayer)

A CPU path tracer written in Rust. It renders objects in a physically-based way using the Cook-Torrance microfacet model.

Prayer is blazing fast:

- It uses Rust's exceptional multi-threading capabilities for concurrent rendering, taking full advantage of multi-core hardware.
- It stores meshes in a K-D Tree acceleration structure, making it cheap to render meshes with a high polygon count.

Some other features include HDRI environment maps, a unified material representation, and an intuitive scene file format.
{% end %}

{% card(image="/projects/oxide.png", alt="An example program written in Oxide") %}

## [Oxide](https://github.com/CRefice/oxide)

A small, general-purpose imperative programming language written with simplicity in mind.
Programs written in Oxide are compiled to run on a custom VM with minimal runtime overhead and memory footprint.

One of my favorite features in modern scripting languages, first-class functions, is fully supported, allowing (and encouraging) functional-style code.
{% end %}

{% card(image="/projects/ssm.png", alt="An example program written in Oxide") %}

## [SSM - Speedy SIMD Math](https://github.com/CRefice/ssm)

SSM is a C++ linear algebra library for use in 3D rendering applications.
It includes types and functions to perform vector transformations in any number of dimensions, using matrices of any size.

Originally written for learning purposes, SSM sports a similar API to [GLM](https://glm.g-truc.net), but with a few key differences:

- All vector and matrix types are generic over type and size, and their dimensionality is known at compile time. This makes it impossible to, for example, multiply together matrices of the wrong size.
- Wrapper types for unit vectors and quaternions allow you to express whether a vector is expected to be normalized at call sites through semantic typing, avoiding unnecessary normalizations and minimizing programmer error.
- Performance: all vector and matrix operations are heavily optimized using SIMD intrinsics, for maximum runtime performance.

{% end %}

{% card(image="/projects/damage.png", alt="A screenshot of Damage running The Legend of Zelda: Link's Awakening.") %}

## [Damage](https://github.com/CRefice/Damage)

A GameBoy emulator for Windows. It's my first ever "real" programming project born out of my love of emulation, and as such contains some more advanced/unusual features such as full audio emulation, integer display scaling, and input remapping.

{% end %}
