===============================================================================
# v.20241021 of FCHECK4 perl program
FCHECK4 User Manual


1. Introduction
2. Requirements
3. How to Run
4. How to Configure
5. Extending FCHECK

===================
Introduction
===================
FCHECK is used to validate SeaBASS data files in order to better standardize
them.  FHCECK checks the format of the header block and quality checks all of
the fields in the data block to minimize the possibility of erroneous data being
archived.  FCHECK has been re-written from scratch to improve stability, 
efficiency, maintainability, and functionality, as well as to allow users to
download FCHECK and use it locally, rather than having to rely on our remote
services.

The current author and maintainer is David Norris.  For any questions,
comments, or suggestions, you can email him/me at david.j.norris@nasa.gov.

If anything breaks or an error is thrown that shouldn't or one isn't thrown
that should be, email the author what happened and the offending data file.

===================
Requirements
===================
FCHECK itself runs on >= Perl v5.10.  It should probably run on older versions,
but it has never been tested.  FCHECK has zero external dependencies, not even
core libs.  

The bathymetry modules, however, require Perl version 5.10 or later and depend
on List::Utils and Fcntl, both of which should be included with the default
Perl installation.  If Perl v5.10 is not available, they are automatically
disabled and a message is added to the report (provided that the message isn't
set to ignore, but more on that later). 

===================
How to Run
===================
FCHECK should run out of the box with no configuration.  To run FCHECK from
the command line:
	
fcheck.pl [-i INI_FILE] FILE...
	If given any directories, all files within will be recursively checked.
	

===================
How to Configure
===================
FCHECK has out grown simple look-up tables and now requires a full blown
configuration file.  This allows FCHECK to be very easily updated by people
with little or no programming experience.  Not everything is accounted for
(more on this later), but I did what I could.

Below are the sections you'll find in the config file, fcheck.ini, and how
to use them.  Each section is merely a pipe (|) delimited table.  The order
used below may not be the actual order in the config file.

The config file _should_ not be case sensitive, but using lowercase will be
more reliable and uniform.

--------------
[general]
--------------
This section is laid out only as (name|value) pair and currently only has one
variable.

bathymetry_check | (srtm30,etopo2,etopo1,etopo1_ice,etopo1_bed,getasse30,globe,none)
                   ** FCHECK is no longer distributed with the bathymetry modules
                      so this should be set to none

Each dataset is slightly different.  Some are better suited than others.
SRTM30 : Actually SRTM30_PLUS, which is a 30" resolution DEM dataset with
         bathymetry, gathered from various sources.
         Default location: datasets/topo30
         http://topex.ucsd.edu/WWW_html/srtm30_plus.html
         
ETOPO2 : NOAA's 2' resolution DEM and bathymetry model, version 2.
         Default location: datasets/ETOPO2v2c_i2_LSB.bin
         http://www.ngdc.noaa.gov/mgg/fliers/01mgg04.html

ETOPO1 : NOAA's 1' resolution DEM and bathymetry model. Two versions are
         available, ice surface and bedrock.  Ice surface is default if neither
         is given.  To specify one, use etopo1_bed or etopo1_ice.
         Default location: datasets/etopo1_<type>_c_i2.bin
         http://ngdc.noaa.gov/mgg/global/global.html

GETASSE30 : A dataset by Marc Bouvet used by default by BEAM.  The height given
            by this set is relative to the WGS84 reference ellipsoid.  As this
            has no defined sea level, one has been approximated based on 
            setting the bar so that about 2/3 of the points were below it.  The
            default is 50 meters, but may be changed by adding _<meters> to the
            VALUE, ie: getasse30_100 will set the bar to 100m.
            Default directory: datasets/getasse30/
            http://www.brockmann-consult.de/beam/doc/help/visat/GETASSE30ElevationModel.html
				(To download go to the link below, Downloads > Software >  GETASSE30 DEM)
				http://www.brockmann-consult.de/cms/web/beam/welcome

GLOBE : NOAA's 1-km DEM model.  Bathymetry is not included, so water has a
        constant value of -500.
        Default directory: datasets/globe/
        http://www.ngdc.noaa.gov/mgg/topo/globe.html
        
none : Don't check if the locations are in water.
        
The main differences between these datasets are the resolutions and the 
possibility to linearly interpolate the values.  SRTM30 and ETOPO1/2 have
bathymetry included, so interpolation is easy and pretty accurate. GLOBE
uses interpolation, but with every water being -500, it very slightly favors
water (first results below).  GETASSE30 is just plain difficult to deal with.

One other main difference, which doesn't affect accuracy, is whether the sets
are available as one, large binary file or as tiles.  GETASSE30 and GLOBE are
only available as tiles.  This makes the resulting measurement take slightly
more time than the other datasets. (More results below.)

All of the following results were at 0.1 degree resolution for the entire globe:

Points in water: 4233352/6480000 getasse30_50 (or getasse30 < 50m)
Points in water: 4356033/6480000 globe < 0
Points in water: 4288702/6480000 srtm30_plus < 0
Points in water: 4298076/6480000 etopo2 < 0
Points in water: 4292282/6480000 etopo1_ice < 0

The following results are the times it took to create a 0.1 degree resolution
raster for the entire globe, all run back-to-back on a standard Linux desktop.
(A total of 6480000 measurement checks for each dataset and some I/O ops.)

Elapsed time for ETOPO2: 264.951168 seconds
Elapsed time for ETOPO1_bed: 282.094715 seconds
Elapsed time for GLOBE: 440.86316 seconds
Elapsed time for GETASSE30: 609.425875 seconds
Elapsed time for SRTM30_PLUS: 278.15055 seconds
Elapsed time for ETOPO1_ice: 267.939873 seconds

Also, bearing in mind that most data files will only have a few dozen or, at
most, a few hundred bathymetry checks, calculation times are almost negligible.

FCHECK is distributed with an empty folder in its directory called datasets/.
When downloading the binary file(s) for a dataset, make sure the file name 
matches up to what FCHECK expects.  If the dataset is split into tiles, drop 
every tile, as is, into the folder FCHECK expects, without any subdirectory.  
For example, ETOPO1 comes in a variety of formats, but FCHECK will only read 
cell-registered, binary files that are 2-byte integer little-Endian for the
ETOPO1 sets, so the desired filename is etopo1_<type>_c_i2.bin, not 
etopo1_<type>_g_i2.bin.

At the time of writing this, these are the links to download the right files:

ETOPO1: 
    bedrock: http://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/cell_registered/binary/etopo1_bed_c_i2.zip
    ice: http://ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/cell_registered/binary/etopo1_ice_c_i2.zip
    
ETOPO2:
    http://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/raw_binary/ETOPO2v2c_i2_LSB.zip
    
GETASSE30: 
    http://www.brockmann-consult.de/cms/web/beam/dlsurvey?p_p_id=downloadportlet_WAR_beamdownloadportlet10&what=data/GETASSE30.zip
    
SRTM30_PLUS: 
    ftp://topex.ucsd.edu/pub/srtm30_plus/topo30/topo30
    
GLOBE:
    http://www.ngdc.noaa.gov/mgg/topo/DATATILES/elev/all10g.zip
 or
    http://www.ngdc.noaa.gov/mgg/topo/DATATILES/elev/all10g.tgz

----------------------
[header_value_comparison_problem] (previously named [header_compares])
----------------------
This section is provided to compare header values to each other.  

FIELD NAME 1 and 2 must be header names, with the starting /, and must match 
a header exactly, case insensitive.  

COMPARE is how to compare the two values, and can
be any of the following: ==, !=, >, <, >=, or <= for numerical compares, eq,
ne, gt, lt, ge, or le for string comparisons.  (String comparisons don't parse
the values at all, so, for example, 1.0 does not equal 1.)  

WARNING and ERROR will override the default report message and severity.
If no warning or error is given, an error of _header_compares is assumed.

--------------
[strings]
--------------
This section is a list of all the possible error or warning messages that can
be reported.

NAME is the name referenced by other sections of the config file, as well as
hard-coded into FCHECK's inner workings.

SEVERITY is where the string will be reported by default.  2 will be an error,
1 will be a warning, and 0 means the string will be ignored.

STRING is what will be displayed in the report.  Anything in curly brackets
will be replaced with its respective value.  There's no real pattern to what
each message will be given, unfortunately.  If you're overriding default
strings in other sections of the config file, use the default as a template.

-------------
[fields]
-------------
This section defines all the valid data fields that FCHECK understands.  It
tells FCHECK how to check each field.

FIELD is the name of the field, without any wave length that may show up in the
real file.  For example, Lu510.0 should be represented as Lu.

UNIT is a comma-separated list of valid units for each field.  For example, the
depth may be m,meters,in,inches.

PARSE AS is a list of modifiers to further validate the field.  The complete
list of possible values is below.

non_null : The value cannot be one of the values listed in /missing
pos(itive)? : The value is numerically checked for a lower bound of 0.
neg(ative)? : The value is numerically checked for an upper bound of 0.
date : The value is parsed as YYYYMMDD or YYYYJJJ.*
time : The value is parsed as HH:MM:SS and each part is validity checked.*
year : The value is parsed as an integer.*
int(eger)? : The value is parsed as an integer, decimal values will result in
             an error.
string : The value is parsed as a string and no numerical checks are given.
float : The value is parsed as an integer, decimal, or scientific notation.

*date, time, and year are special in that their upper bound can be 'today',
    which will calculate today's date/time/year and use that.
    
If date, time, year, integer, string, or float are not specified, float is
assumed.

If a value cannot be parsed as desired, _field_failed_to_parse is reported.

LOWER and UPPER BOUND, if present, are used to determine the range of valid
values.

WARNING and ERROR will override the default report message and severity for
the bounds checking only.  If no warning or error is given, an error of 
_field_out_of_bounds is assumed for integers and floats.  Dates and times get
a slew of their own error messages that are beyond the scope of this document.

-------------
[suffixes]
-------------
This section is a newline-separated list of suffixes that can appear at the end
of any field name.  For example, a field for chlorophyll standard deviation
would be chl_sd.  The field must exist in order for the suffix to be valid.
The unit of the field with the suffix must match the unit of the field without
it.

--------------
[headers]
--------------
This section defines every valid header understood by FCHECK.

HEADER is the name of the header, with the starting /.

REQUIRED is one of the following: required, optional, or obsolete. It can also
have a modifier of no_value, which means it is not expected to have a =<value>
following the header name.  If a required header is not found, an error, 
_required_header_not_found will be thrown.  If an option header is not found,
a warning, _optional_header_not_found, is thrown.  If a header is not found on
the list at all, _unknown_header will be thrown.  Obsoletes will be silently
ignored unless WARNING or ERROR is overridden.

WARNING and ERRORs are thrown if a required or optional header aren't present.
They are also thrown if an obsolete header is found in conjunction with an
appropriate warning message like 'obsolete_parameters_field_detected'

'obsolete' does not generate WARNING or ERRORS when not used in conjunction with 
a specific message. Because of its quiet nature, it is repurposed to label certain
headers that are considered optional or only situationally required (i.e., it
suppresses warning messages). In a future version of FCHECK a different label
might be implemented to avoid ambiguity or confusion within the .ini file.

--------------
[numbers]
--------------
This section allows headers to be parsed and checked.

HEADER is the header name, exactly as it is found, or part of the header name.
For example, you could check /north_latitude and /south_latitude, or you could
put simply latitude and it will find and check them both.

TYPE is similar to PARSE AS in [fields] with two differences.  Header times,
latitudes, and longitudes must have unit labels, [GMT] and [DEG].  And, the
TYPE may be plural, which is considered a comma-separated list of values of
the given type.  TYPE may also contain non_null, meaning the header can't be
missing.

---------------
[validity]
---------------
This section allows headers to checked for valid or invalid values.

HEADER is the header name, exactly as it is found, or part of the header name.
For example, you could check /north_latitude and /south_latitude, or you could
put simply latitude and it will find and check them both.  HEADER may also be
an asterisk (*) to check every header.

(IN)VALID may be either valid or invalid.  If set to invalid and the line
evaluates to true, WARNING/ERROR is thrown.  If set to valid, and the line 
evaluates to false, WARNING/ERROR is thrown.

If neither WARNING or ERROR is present, _validity_failed is assumed.

MODIFIERS define how to interpret the list of VALUES.  The complete list of
options is below.

equals : true if value equals VALUES
contains : true if value contains VALUES
exact : true if value equals/contains VALUES, exactly as written
any : true if value equals/contains any of the delimited VALUES 
all : true if value equals/contains all of the delimited VALUES 
float : force a float comparison
int : force an integer comparison

'any' and 'all' FCHECK  that VALUES is a list of values to check. If followed 
by a non-space character, that character is used as the delimiter.  Default 
is a comma.

To avoid confusion, equals or contains should be specified, and one of exact,
any, or all.  equals and exact are assumed.

VALUES can be a number, string, or a header (with preceding /), or a mixed list
of the three.

-------------
[report]
-------------
This section modifies the way report is given.  It can modify the general look
or the look of specific error messages.

ERROR_NAME is either an asterisk (*) or the name of the error as it appears in
[strings].  * modifies every warning/error.

HEADER is the name of a string to put before the error/warning section for that
error/warning.  FOOTER is put at the end.

MODIFIER is how to change the error given.  The complete list of values is
shown below.

truncate : Limits the times the error is printed.  If followed by a number,
           that number is used as the max lines to print. Default: 5.
skip     : Removes the error from the report.
ignore   : Same as skip.
warning  : Forces the error to be reported as a warning.
error    : Forces the warning to be an error.
split    : Each error gets its own line instead of being clumped together.
           Messages split this way are still tallied the same in the error and
           warning counts.
summary  : Used to only print errors once for each occurrence.  Any bracketed
           names in the HEADER are replaced with a delimited list of the unique
           values of that type.  Same goes with FOOTER, except that each unique
           set gets its own line.  If followed by a character, that character
           is used as the delimiter. Default: <space>.
wrap     : Wraps the line to a reasonable width, breaking on word boundaries.
           If followed by a number, that number is used as the width (default: 
           90).  Wrapping is the very last step in the output.  It won't be 
           wrapped exactly to the width, as various forms of spacing are added
           to make it more readable.
           
The global MODIFER column (line with ERROR_NAME of *) can have special nulling
values.  Simply add no_<modifier you wish to disregard>, such as no_split or
no_summary, and any column with the given modifier will be removed at run-time.


===================
Extending FCHECK
===================
There is a place specifically for non-config-file driven checks.  It's the very
last subroutine in the code, all the way at the bottom, called one_offs.  This
is where any additional checks can be put in.  The instructions and such are
right above the function itself.  Best of luck to you.
