#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2012,2013,2014, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

if(SOURCE_IS_SOURCE_DISTRIBUTION)
    # Make sure source package contains all man pages.
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/man1/gmx-view.1")
        message(FATAL_ERROR "Man pages are missing from source package.")
    endif()
endif()

add_custom_target(man
    ${CMAKE_COMMAND} -E make_directory man1
    COMMAND ${CMAKE_COMMAND} -E make_directory man7
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/man7/gromacs.7.in" man7/
    COMMAND gmx -quiet help -export man
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating man pages"
    VERBATIM)

set(MAN_PAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
if (GMX_BUILD_HELP)
    set_target_properties(man PROPERTIES EXCLUDE_FROM_ALL OFF)
    set_directory_properties(PROPERTIES
        ADDITIONAL_MAKE_CLEAN_FILES "man1;man7/gromacs.7")
    set(MAN_PAGE_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# man pages are only available if they are either build or this is a source archive
if (SOURCE_IS_SOURCE_DISTRIBUTION OR GMX_BUILD_HELP)
    install(FILES ${MAN_PAGE_DIR}/man7/gromacs.7
        DESTINATION ${MAN_INSTALL_DIR}/man7
        COMPONENT man)
    install(DIRECTORY ${MAN_PAGE_DIR}/man1
        DESTINATION ${MAN_INSTALL_DIR}
        COMPONENT man)
endif()
