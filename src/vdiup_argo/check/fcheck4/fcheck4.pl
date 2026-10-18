#!/usr/bin/env perl
# v.20241021 of FCHECK4 perl program

# The FCHECK code is in the public domain, available without fee for educational, research, non-commercial and 
# commercial purposes. Users may distribute this code to third parties provided that this statement appears on
# all copies and that no charge is made for such copies.
# 
# NASA GSFC MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THE SOFTWARE FOR ANY PURPOSE. IT IS PROVIDED 
# "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY. NEITHER NASA GSFC NOR THE U.S. GOVERNMENT SHALL BE LIABLE FOR
# ANY DAMAGE SUFFERED BY THE USER OF THIS SOFTWARE.

package fcheck4;

use strict;
use warnings FATAL => 'all';
use Data::Dumper;

$Data::Dumper::Sortkeys = 1;
if (-l __FILE__){
    $0 = readlink(__FILE__);
}

#use Time::HiRes;
#my $start = [ Time::HiRes::gettimeofday( ) ];

our $dirname = dirname($0);
my $ini_file = "$dirname/fcheck.ini";

strip(@ARGV);

if ($#ARGV < 0 || ($#ARGV <= 1 and $ARGV[0] eq "-i")) {
	print(usage());
	exit(1);
}
## 20240401 find -i flag  and handle the subsequent filepath independent of the order in which arguments are provided

for (my $i = 0; $i <= $#ARGV; $i++) {
    if ($ARGV[$i] eq "-i") {
        if ($i < $#ARGV) {
            # Assign the subsequent argument to $ini_file
            $ini_file = $ARGV[$i + 1];
            
            # Remove both "-i" and the subsequent argument from @ARGV
            splice(@ARGV, $i, 2);
            last;  # Stop searching after finding "-i"
        } else {
            die "No argument provided after -i flag.";
        }
    }
}

# Handle "-env" flag for fchecking env files
# This could be set up to use the -env flag to use our default *env.ini file rather than users needing to specify the filepath with -i
my $env_arg = 0;
for (my $i = 0; $i <= $#ARGV; $i++) {
    if ($ARGV[$i] eq "-env") {
        $env_arg = 1;
        splice(@ARGV, $i, 1);  # Remove "-env" from @ARGV
        last;  # Stop searching after finding "-env"
    }
}

if($env_arg == 1 && $ini_file ne "$dirname/fcheck.ini"){
	print " The -i and and -env flags should not both be used at once to specific a configuration file.\n";
	exit 1;
}

# Assign env configuration file to $ini_file
if ($env_arg == 1){
	if (-e "$dirname/fcheck_env.ini") {
		$ini_file = "$dirname/fcheck_env.ini";
		print "\nAttempting to check files using fcheck_env.ini built for ENV files\n\n";
	} else {
		print "The fcheck_env.ini file is not present in the directory that fcheck is running from\n";
		exit 1;
	}
}

our %headers;
our %errors;
our %warnings;
our %config;

our $data_begin;
our @months      = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31);
our @months_leap = (0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31);

our $MAXIMUM_BAD_STARTING_LINES = 5;
our $WL = qr/\d{3,5}(?:\.\d+)?/;

if (-e $ini_file){
	read_config($ini_file);
} else {
	print "\n Configuration (.ini) file " . $ini_file . " not present\n\n"; 
	exit 1;
}


eval {
	if (unpack("s>", pack("s>", 10000)) != 10000) {
		#if block provided just to bypass void context warning
		#but if it did get in here, that means something's broken
	} ## end if (unpack("s>", pack(...
};

our ($bathymetry_warning, $bathymetry_name, $bathymetry_function, $bathymetry_location, $getasse30_sea_height);

if ($@) {
	$bathymetry_warning = 1;
}

check_bathymetry_method();
glob_directories_in_args();

our (@lines, $all_fields, @fields, @missing, $data_delimiter, $bad_file);

our ($final_report, %all_errors, %all_warnings) = ('');
my $total_errors   = 0;
my $total_warnings = 0;
my $file_i = 1;

our $input_filename;
foreach (@ARGV) {
	$input_filename = $_;
	eval {
		if (basename($input_filename) =~ /\s+/) {
			report('spaces_in_filename', filename => $input_filename);
		}
		my $text = join("",slurp($input_filename));
		if ($text =~ /\r/s){
			$text =~ s/\r\n?/\n/sg;
			report('file_in_dos_format');
		}
		if ($text =~ /^\s*[^!].*?,+$/m){
			$text =~ s/([^,]+),+\n/$1\n/mg;
			report('file_in_csv_format_with_extra_commas');
		}
		@lines = split(/\n/,$text);

		$bad_file = 0;

		read_header();

		if (!$bad_file) {
			$all_fields = $config{'fields'};
			@fields     = (defined($headers{'/fields'}) ? split(",", $headers{'/fields'}) : ());
			@missing    = (defined($headers{'/missing'}) ? split(",", $headers{'/missing'}) : ());
			

			if (defined($headers{'/below_detection_limit'})){
				push(@missing, split(",", $headers{'/below_detection_limit'}));
			}

			if (defined($headers{'/above_detection_limit'})){
				push(@missing, split(",", $headers{'/above_detection_limit'}));
			}

			push(@missing, "na", "none", "n/a");

			if (defined($headers{'/delimiter'})){
				$data_delimiter = $headers{'/delimiter'};
				if ($data_delimiter =~ /comma/i) {
					$data_delimiter = ',';
				} elsif ($data_delimiter =~ /space/i) {
					$data_delimiter = '\s+';
				} elsif ($data_delimiter =~ /tab/i) {
					$data_delimiter = "\t";
				} elsif ($data_delimiter =~ /semi-?colon/i) {
					$data_delimiter = ';';
				} else {
					$data_delimiter = undef;
				}
			} else {
				$data_delimiter = undef;
			}

			remove_whitespace_lines_at_end();
			
			if ($bathymetry_warning){
				report('bathymetry_warning_version_number');
			}

			## Check if header key and value contain non-ascii chars
			foreach my $key ( keys %headers ) {
				my $value = $headers{$key};
				
				if ( $key =~ /[^[:ascii:]]/i ) {
					report("non_ascii_char_in_header", header => $key);
				} elsif ( $value =~ /[^[:ascii:]]/i ) {
					report("non_ascii_char_in_header_val", header => $key, value => $value);
				}
			}

			check_headers_for_whitespace();
			check_for_required_headers();
			check_for_unknown_headers();
			check_for_invalid_numbers();
			check_header_compares();
			check_validity_section();
			check_fields();
			check_data();
			one_offs();

		} ## end if (!$bad_file)

		make_report();

		$total_errors   += scalar(keys(%errors));
		$total_warnings += scalar(keys(%warnings));

		%headers  = ();
		%errors   = ();
		%warnings = ();
	};
	if ($@){
		my $err = $@;
		print "Error while processing $input_filename, $err\n";
	}
	$file_i++;
} ## end foreach $input_filename (@ARGV)

#if ($#ARGV > 0) {
	print '#'x100 . "\n";
	if (%all_errors || %all_warnings){
		#printf("SCAN SUMMARY:\n\nDifferent types of issues detected during the scan: %i error%s and %i warning%s\n\n", $total_errors, ($total_errors == 1 ? '' : 's'), $total_warnings, ($total_warnings == 1 ? '' : 's'));
		printf("SCAN SUMMARY\n\n");
		if (%all_errors){
			my $summary_string = join(', ', map {
				my $e = $_;
				my $fc = keys(%{$all_errors{$e}});
				my $wc = 0;
				while (my ($k, $v) = each(%{$all_errors{$e}})){
					$wc += $v;
				}
				"$_\[$fc\]\($wc\)"
			} sort(keys(%all_errors)));
			print join("", "Types of errors detected:\n\t", wrap($summary_string, 100));
			print "\n";
		}
		if (%all_warnings){
			my $summary_string = join(', ', map {
				my $e = $_;
				my $fc = keys(%{$all_warnings{$e}});
				my $wc = 0;
				while (my ($k, $v) = each(%{$all_warnings{$e}})){
					$wc += $v;
				}
				"$_\[$fc\]\($wc\)"
			} sort(keys(%all_warnings)));
			print join("", "Types of warnings detected:\n\t", wrap($summary_string, 100));
			print "\n";
		}
		printf("Format of listed problems: problem-name[number-of-files-affected](total-number-of-occurrences)\n\n");
	} else {
		print "No errors or warnings reported.\n";
	}
#} ## end if ($#ARGV > 0)

#my $elapsed = Time::HiRes::tv_interval( $start );
#print "Elapsed time: $elapsed seconds!\n";

exit(0);

#--------------------------------------------------------------------------------------------------
# usage()
#	Prints the usage of this program
#--------------------------------------------------------------------------------------------------
sub usage {
	my $usemsg = <<"EndUsage";
Usage: 
Command line: fcheck4.pl [-i INI_FILE] FILE...
	If given any directories, all files within will be recursively checked.
EndUsage
	return ($usemsg);
} ## end sub usage

#--------------------------------------------------------------------------------------------------
# glob_directories_in_args()
#	Runs through the ARGs. If any one is a directory, remove it and add all files it contains.
#	If any files were directories, do the same to them, recursively adding the entire tree.
#	Removes any duplicates found, as well.
#--------------------------------------------------------------------------------------------------
sub glob_directories_in_args {
	for (my $i = 0; $i <= $#ARGV; $i++) {
		if (-d $ARGV[$i]) {
			my $dir = $ARGV[$i];
			$dir =~ s'/$'';
			splice(@ARGV, $i, 1, glob("$dir/*"));
			$i--;
		} elsif ($ARGV[$i] =~ /\*/){
			my $glob = $ARGV[$i];
			splice(@ARGV, $i, 1, glob("$glob*"));
			$i--;
		}
	} ## end for (my $i = 0; $i <= $#ARGV...
	@ARGV = uniq(@ARGV);
} ## end sub glob_directories_in_args

#--------------------------------------------------------------------------------------------------
# uniq(@array)
#	Returns a list of all the unique values in an array, order preserved.
#--------------------------------------------------------------------------------------------------
sub uniq {
	my @arr = @_;
	my %seen;
	my @unique = grep {!$seen{$_}++} @arr;
	return @unique;
} ## end sub uniq

#--------------------------------------------------------------------------------------------------
# check_bathymetry_method()
#   Checks the settings file for the bathymetry setting. If it exists, it points $bathymetry_name,
#		$bathymetry_function, and $bathymetry_location to the right values for is_in_water to use.
#
#	This should not be called unless unpack("s>",$val) is a valid expression.  This requires Perl
#		version 5.10.
#
# If the desired method is GETASSE30, $getasse30_sea_height is set, as well.  If not defined,
#	$getasse30_sea_height will be set to 50
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#	our ($bathymetry_name, $bathymetry_function, $bathymetry_location, $getasse30_sea_height)
#		has been defined.
#--------------------------------------------------------------------------------------------------
sub check_bathymetry_method {
	eval {
		my %gen_config = %{$config{'general'}};
		if (defined($gen_config{'bathymetry_check'})) {
			my $b = lc($gen_config{'bathymetry_check'});
			if ($b eq "etopo1" or $b eq "etopo1_ice") {
				require SeaBASS::DEM::ETOPO1;
				SeaBASS::DEM::ETOPO1->import(qw(noaa_etopo1));
				$bathymetry_function = \&noaa_etopo1;
				$bathymetry_location = "$dirname/datasets/etopo1_ice_c_i2.bin";
				$bathymetry_name     = "etopo1";
			} elsif ($b eq "etopo1_bed") {
				require SeaBASS::DEM::ETOPO1;
				SeaBASS::DEM::ETOPO1->import(qw(noaa_etopo1));
				$bathymetry_function = \&noaa_etopo1;
				$bathymetry_location = "$dirname/datasets/etopo1_bed_c_i2.bin";
				$bathymetry_name     = "etopo1";
			} elsif ($b eq "etopo2") {
				require SeaBASS::DEM::ETOPO2;
				SeaBASS::DEM::ETOPO2->import(qw(noaa_etopo2));
				$bathymetry_function = \&noaa_etopo2;
				$bathymetry_location = "$dirname/datasets/ETOPO2v2c_i2_LSB.bin";
				$bathymetry_name     = "etopo2";
			} elsif ($b eq "globe") {
				require SeaBASS::DEM::GLOBE;
				SeaBASS::DEM::GLOBE->import(qw(noaa_globe));
				$bathymetry_function = \&noaa_globe;
				$bathymetry_location = "$dirname/datasets/globe";
				$bathymetry_name     = "globe";
			} elsif ($b eq "srtm30_plus" or $b eq "srtm30") {
				require SeaBASS::DEM::SRTM30_PLUS;
				SeaBASS::DEM::SRTM30_PLUS->import(qw(ucsd_srtm30_plus));
				$bathymetry_function = \&ucsd_srtm30_plus;
				$bathymetry_location = "$dirname/datasets/topo30";
				$bathymetry_name     = "srtm30_plus";
			} elsif ($b =~ /getasse(30)?_?(.*?)$/) {
				require SeaBASS::DEM::GETASSE30;
				SeaBASS::DEM::GETASSE30->import(qw(beam_getasse30));
				$bathymetry_function = \&beam_getasse30;
				$bathymetry_location = "$dirname/datasets/getasse30";
				$bathymetry_name     = "getasse30";
				if ($1) {
					$getasse30_sea_height = $1;
				} else {
					$getasse30_sea_height = 50;
				}
			} elsif ($b ne "none") {
				report('config_file_error', message => "Bathymetry not recognized. Bathymetry will not be checked.");
			}
		} ## end if (defined($gen_config...
	};
	if ($@){
		$bathymetry_name = '';
	}
} ## end sub check_bathymetry_method

#--------------------------------------------------------------------------------------------------
# make_report([$output_file_handle])
#   Creates and prints a report based in the errors and warnings found
#   Relies heavily on the [report] section of the config file and does the actual parsing,
#       modifying %errors and %warnings accordingly
#	If given a file handle, the report will be printed to it.  Else, to STDOUT.
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#--------------------------------------------------------------------------------------------------
sub make_report {
	my $output_file_h = shift;

	my %report_config = %{$config{'report'}};

	my $error_count   = 0;
	my $warning_count = 0;

	while (my ($error_name, $cols_ref) = each(%report_config)) {
		my ($modifiers, $header, $footer) = @{$cols_ref};
		if (not $modifiers) {
			next;
		}
		if ($modifiers =~ /skip|ignore/i) {
			if ($error_name eq "*") {
				%errors   = ();
				%warnings = ();
			} else {
				if ($errors{$error_name}) {
					delete $errors{$error_name};
				}
				if ($warnings{$error_name}) {
					delete $warnings{$error_name};
				}
			} ## end else [ if ($error_name eq "*")
		} else {
			if ($modifiers =~ /warning/i) {
				if ($error_name eq "*") {
					while (my ($error_name, $errs_ref) = each(%errors)) {
						if ($warnings{$error_name}) {
							push(@{$warnings{$error_name}}, $errors{$error_name});
						} else {
							$warnings{$error_name} = $errors{$error_name};
						}
					} ## end while (my ($error_name, $errs_ref...
					%errors = ();
				} elsif ($errors{$error_name}) {
					if ($warnings{$error_name}) {
						push(@{$warnings{$error_name}}, $errors{$error_name});
					} else {
						$warnings{$error_name} = $errors{$error_name};
					}
					delete $errors{$error_name};
				} ## end elsif ($errors{$error_name...
			} elsif ($modifiers =~ /error/i) {
				if ($error_name eq "*") {
					while (my ($error_name, $errs_ref) = each(%warnings)) {
						if ($errors{$error_name}) {
							push(@{$warnings{$error_name}}, $warnings{$error_name});
						} else {
							$errors{$error_name} = $warnings{$error_name};
						}
					} ## end while (my ($error_name, $errs_ref...
					%warnings = ();
				} elsif ($warnings{$error_name}) {
					if ($errors{$error_name}) {
						push(@{$errors{$error_name}}, $warnings{$error_name});
					} else {
						$errors{$error_name} = $warnings{$error_name};
					}
					delete $warnings{$error_name};
				} ## end elsif ($warnings{$error_name...
			} ## end elsif ($modifiers =~ /error/i)
		} ## end else [ if ($modifiers =~ /skip|ignore/i)

	} ## end while (my ($error_name, $cols_ref...

	while (my ($error_name, $errors) = each %errors) {
		$error_count += scalar(@{$errors});
	}
	while (my ($error_name, $warnings) = each %warnings) {
		$warning_count += scalar(@{$warnings});
	}

	my $error_plural   = "";
	my $warning_plural = "";
	my $verb_to_be     = "was";

	if ($error_count != 1) {
		$error_plural = "s";
	}
	if ($warning_count != 1) {
		$warning_plural = "s";
		$verb_to_be     = "were";
	}

	my $report = '#'x100 . "\n";
	if (@ARGV > 1){
		$report .= "File $file_i: $input_filename";
	} else {
		$report .= $input_filename;
	}

	if ($error_count == 0) {
		$report .= "\n\nThis file passed the FCHECK.\n\n";
	} else {
		$report .= "\n\nThis file failed the FCHECK.\n\n";
	}

	#$report .= sprintf(($output_file_h ? $output_file_h : \&STDOUT),"%i error%s (%i unique) and %i warning%s %s found.\n\n", $error_count, $error_plural, scalar(keys(%errors)), $warning_count, $warning_plural, $verb_to_be );
	$report .= sprintf("%i error%s (%i unique) and %i warning%s (%i unique) %s found.\n", $error_count, $error_plural, scalar(keys(%errors)), $warning_count, $warning_plural, scalar(keys(%warnings)), $verb_to_be);
	if ($error_count) {
		$report .= "\n******************************************** ERRORS ***********************************************\n";
		$report .= report_hash(\%errors);
	}

	if ($warning_count) {
		$report .= "\n******************************************* WARNINGS **********************************************\n";
		$report .= report_hash(\%warnings);
	}

	$report .= "\n";

	if ($output_file_h) {
		print $output_file_h $report;
	} else {
		print $report;
	}
} ## end sub make_report

#--------------------------------------------------------------------------------------------------
# report_hash(%hash)
#   Given %errors/%warnings, creates and prints a report of the contents
#
# Pre-conditions:
#   read_config has been called
#--------------------------------------------------------------------------------------------------
sub report_hash {
	my $ret           = "";
	my $hash_ref      = shift;
	my %report_config = %{$config{'report'}};
	my $cur_error     = 0;

	my $global_modifiers = "";
	if (defined $report_config{"*"}) {
		$global_modifiers = @{$report_config{"*"}}[0];
	}

	while (my ($error_name, $errors_ref) = each %$hash_ref) {
		$cur_error += 1;
		my @errors = @$errors_ref;

		my $output;
		my ($modifiers, $header, $footer);
		my $note = undef;

		if (defined $report_config{$error_name}) {
			($modifiers, $header, $footer) = @{$report_config{$error_name}};
		}

		if (not $modifiers) {
			$modifiers = "";
		}
		if (defined $modifiers or $global_modifiers) {
			if ($global_modifiers) {
				$modifiers .= " $global_modifiers";
			}

			if ($modifiers !~ /no_(skip|ignore)/ and $modifiers =~ /skip|ignore/) {
				@errors = ();
			} elsif ($modifiers !~ /no_truncate/ and $modifiers =~ /truncate(\d*)/ and ($modifiers =~ /no_summary/ or $modifiers !~ /summary/)) {
				my $count = $1;
				if (not defined($count)) {
					$count = 5;
				}

				if ($count <= $#errors) {
					@errors = @errors[0 .. min($count - 1, $#errors)];
					if ($count > 0) {
						$note = "...";
					}
				} ## end if ($count <= $#errors)
			} ## end elsif ($modifiers !~ /no_truncate/...

			if ($note) {
				push(@errors, $note);
			}
		} ## end if (defined $modifiers...
		if (defined $report_config{$error_name}) {
			if ($modifiers =~ /no_summary/ and $modifiers =~ /\bsummary/) {
				$header = $footer = undef;
			} else {
				if ($header and $config{'strings'}{$header}) {
					$header = @{$config{'strings'}{$header}}[1];
				}
				if ($footer and $config{'strings'}{$footer}) {
					$footer = @{$config{'strings'}{$footer}}[1];
				}
			} ## end else [ if ($modifiers =~ /no_summary/...
		} ## end if (defined $report_config...

		if ($modifiers =~ /split/ and $modifiers !~ /no_split/) {
			$output = "";
			foreach my $error (@errors) {
				if ($output) {
					$cur_error += 1;
				}
				$output .= sprintf("%-4s%s\n", "$cur_error)", $error);
			} ## end foreach my $error (@errors)
		} elsif ($modifiers =~ /summary(.?)/ and $modifiers !~ /no_summary/) {
			$output = "";

			if ($config{'strings'}{$error_name}) {
				my $delim = $1 || " ";

				if ($header) {
					my $message           = $header;
					my @fields_to_replace = ();
					while ($message and $message =~ /(\{.*?\})/) {
						push(@fields_to_replace, $1);
						$message =~ s/$1//;
					}
					$message = $header;
					my $original = @{$config{'strings'}{$error_name}}[1];
					my @lines    = ();
					foreach my $line (@errors) {
						my $original_header = $header;
						foreach my $field_to_replace (@fields_to_replace) {
							if ($original =~ /(^.*?)\Q$field_to_replace\E(.*?$)/) {
								my ($before, $after) = (quotemeta($1), quotemeta($2));
								$before =~ s/\\\{.*?\\\}/.*?/g;
								$after  =~ s/\\\{.*?\\\}/.*?/g;
								if ($line =~ /$before(.*?)$after/) {
									my $replace = $1;
									$original_header =~ s/$field_to_replace/$replace/;
								}
							} ## end if ($original =~ /(^.*?)$field_to_replace(.*?$)/)
						} ## end foreach my $field_to_replace...
						push(@lines, $original_header);
					} ## end foreach my $line (@errors)
					$output .= join('', "    ", join("\n    ", uniq(@lines)), "\n");
				} ## end if ($header)#revised to match footer section
				if ($footer) {
					my $message           = $footer;
					my @fields_to_replace = ();
					while ($message and $message =~ /(\{.*?\})/) {
						push(@fields_to_replace, $1);
						$message =~ s/$1//;
					}
					$message = $footer;
					my $original = @{$config{'strings'}{$error_name}}[1];
					my @lines    = ();
					foreach my $line (@errors) {
						my $original_footer = $footer;
						foreach my $field_to_replace (@fields_to_replace) {
							if ($original =~ /(^.*?)\Q$field_to_replace\E(.*?$)/) {
								my ($before, $after) = (quotemeta($1), quotemeta($2));
								$before =~ s/\\\{.*?\\\}/.*?/g;
								$after  =~ s/\\\{.*?\\\}/.*?/g;
								if ($line =~ /$before(.*?)$after/) {
									my $replace = $1;
									$original_footer =~ s/$field_to_replace/$replace/;
								}
							} ## end if ($original =~ /(^.*?)$field_to_replace(.*?$)/)
						} ## end foreach my $field_to_replace...
						push(@lines, $original_footer);
					} ## end foreach my $line (@errors)
					$output .= join('', "    ", join("\n    ", uniq(@lines)), "\n");
				} ## end if ($footer)
			} ## end if ($config{'strings'}...

		} else {
			if ($header) {
				unshift(@errors, $header);
			}
			if ($footer) {
				push(@errors, $footer);
			}
			$output = join('', sprintf("%-4s", "$cur_error)"), join("\n    ", @errors), "\n");
		} ## end else [ if ($modifiers =~ /split/...

		if ($output) {
			$output =~ s/\\n/\n    /g;
			$output =~ s/\\t/    /g;

			if ($modifiers !~ /no_wrap/ and $modifiers =~ /wrap(\d*)/) {
				my $width = $1 || 100;
				$output = wrap($output, $width);
			}

			$ret .= $output;
		} else {
			$cur_error -= 1;
		}
	} ## end while (my ($error_name, $errors_ref...

	return $ret;
} ## end sub report_hash

#--------------------------------------------------------------------------------------------------
# is_missing($value)
#   Evaluates to true if the value given is a missing value
#--------------------------------------------------------------------------------------------------
sub is_missing {
	my $val = lc(shift);
	if (defined($val)) {
		my $is_string = is_invalid_float($val);
		foreach my $missing_value (@missing) {
			if ($is_string or is_invalid_float($missing_value)) {
				if ($val eq $missing_value) {
					return 1;
				}
			} elsif ($val == $missing_value) {
				return 1;
			}
		} ## end foreach my $missing_value (...
	} ## end if (defined($val))
	return 0;
} ## end sub is_missing

#--------------------------------------------------------------------------------------------------
# check_data()
#	Checks the data for correct values, dates, locations, columns, etc
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#
# Reports:
#   Date doesn't match YYYYMMDD or YYYYJJJ: _invalid_date
#   Month in date isn't between 1 and 12: _invalid_month
#   Day of month doesn't exist: _invalid_day_of_month
#   A date before 1975 is given: _data_pre_1975_detected
#   Time doesn't match HH:MM:SS: _invalid_time
#   Hours, minutes, or seconds out of range: _invalid_time_value
#   Couldn't parse a given float/integer/date/time: _field_failed_to_parse
#   Date/time/float/degree/int out of bounds: _field_out_of_bounds
#   Julian day doesn't exist: _data_invalid_julian
#   Lat/lon coordinates appear to be on land: _data_bathymetry_failed
#   A field marked as non_null is null: _field_cant_be_missing
#	Empty line in data: _empty_line
#	Leading or trailing spaces: _spaces_around_line
#	Data line columns don't match listed fields: _bad_data_line
#	/begin_data or /end_data@? found: _obsolete_data_tags
#--------------------------------------------------------------------------------------------------
sub check_data {
	my @errors;
	my $old_data_tags_reported = 0;
	my $line_number = $data_begin;
	for (my $i = $data_begin; $i <= $#lines;) {
		$line_number++;
		my $line = $lines[$i++];
		if ($line =~ /^!/) {
			report('invalid_comment_location', line => $line_number);
			next;
		} elsif ($line =~ m"/begin_data"i) {
			if (!$old_data_tags_reported) {
				report('obsolete_data_tags');
			}
			$old_data_tags_reported = 1;
			next;
		} elsif ($line =~ m"/end_data"i) {
			if (!$old_data_tags_reported) {
				report('obsolete_data_tags');
			}
			$old_data_tags_reported = 1;
			last;
		} elsif ( $line =~ /[^[:ascii:]]/i ) {
			report('non_ascii_char', line => $line_number);
			last;
		} ## end elsif ($line =~ m"/end_data"i)

		#strip($line);
		
#		if ($data_delimiter){
#			if ($data_delimiter eq ' ') {
#				$line =~ s/\s+/ /g;
#			} elsif ($data_delimiter eq "\t") {
#				$line =~ s/	+/\t/g;
#			}
#		}

		if ($line =~ /^\s*$/) {
			report('empty_line', line => $line_number, block => 'Data');
			$i--;
			splice(@lines, $i, 1);
			next;
		} elsif ($line =~ /^\s+|\s+$/){
			report('spaces_around_line', line => $line_number, block => 'data');
			strip($line);
		} ## end if ($line =~ /^\s*$/)

		$lines[$i - 1] = $line;

		my %data_line = make_data_hash($i, $line);
        if (!%data_line){
#    		my @keys = keys(%data_line);
#    		if ($#keys != $#fields) {
    			report('bad_data_line', line => $line_number);
    			next;
#    		}
        }

		foreach my $orig_field (@fields) {
			my $field_base = lc($orig_field);
			my $field = $field_base;
			my $value = $data_line{$field};

			my ($suffix_unit, $has_suffix, $keep_checking_dims) = ('', 0, 1);
			keys(%{$config{'suffixes'}});
			while (my ($suffix, $info) = each(%{$config{'suffixes'}})){
				# if ($field_base =~ /_$suffix$/){ # converted to the below line 20240403 by MAM to take prefixes and make suffix searches more rigorous
				if ($field_base =~ /^$suffix|$suffix$/){
					$has_suffix = $suffix;
					# $field_base =~ s/_$suffix$//; # converted to the below line 20240403 by MAM to take prefixes
					$field_base =~ s/^$suffix|$suffix$//g;
					$suffix_unit = $info->[0];
					last;
				}
			}
			my $d_re = qr/(?:\d+\.?\d*)/;
			while ($keep_checking_dims){
				$keep_checking_dims = 0;
				foreach my $suffix (@{$config{'nd_fields'}}){
					my $re;
					if ($suffix->[1] == 0){
						$re = qr/_$suffix->[0]$/;
					} elsif ($suffix->[1] == 1){
						$re = qr/_$d_re$suffix->[0]$/;
					} elsif ($suffix->[1] == 2){
						$re = qr/_$suffix->[0]$d_re$/;
					}
					if ($field_base =~ $re){
						$keep_checking_dims = 1;
						$field_base =~ s/$re//;
					}
				}
			}

			my ($unit, $parse_as, $lbound, $ubound, $warning, $error);

			if ($has_suffix){
				if (!defined($value)){
					next;
				}
				$parse_as = 'float';
			} else {
				$field_base =~ s/(?<![-_])\d{3,5}(?:\.\d+)?$//;    #take out wavelength, the ###.# at the end of the field name

				my $arr = $all_fields->{$field_base};
				unless ($arr) {
					next;
				}
				($unit, $parse_as, $lbound, $ubound, $warning, $error) = @$arr;

				if (!defined($value)) {
					if ($parse_as =~ /non_null/) {
						report('field_cant_be_missing', line => $line_number, field => $orig_field);
					}
					next;
				} elsif ($parse_as =~ /string/) {
					next;
				}

				if ($parse_as =~ /pos(itive)?/ and not $lbound) {
					$lbound = 0;
				} elsif ($parse_as =~ /neg(ative)?/ and not $ubound) {
					$ubound = 0;
				}
			}
			$parse_as = trim($parse_as);

			my ($invalid_ret, $val, $parsing_as, $was_whitespace);
			eval {
				if ($parse_as =~ /int(eger)?/){ 
					if (!($invalid_ret = is_invalid_int($value))) {
						$parsing_as = "int";
						$val        = int($value);
					} else {
						if ($invalid_ret == 3){
							$was_whitespace = 1;
						}
					}
				} elsif ($parse_as =~ /float/) {
					if (!($invalid_ret = is_invalid_float($value))) {
						$parsing_as = "float";
						$val = sprintf("%0.32f", $value);
					} else {
						if ($invalid_ret == 3){
							$was_whitespace = 1;
						}
					}
				} elsif ($parse_as =~ /(year|julian|date)/){
					if (!($invalid_ret = is_invalid_int($value))){
						$parsing_as = "int";
						$val        = int($value);
						if ($ubound eq "today") {
							my $tmp_parse_as = $1;
							if ($tmp_parse_as eq "date") {
								$ubound = get_today();
							} elsif ($tmp_parse_as eq "julian") {
								$ubound = get_today_julian();
							} elsif ($tmp_parse_as eq "year") {
								$ubound = get_today_year();
							} else {
								$ubound = undef;
							}
						} ## end if ($ubound eq "today")
					} else {
						if ($invalid_ret == 3){
							$was_whitespace = 1;
						}
					}
				} elsif ($parse_as =~ /time/){
					# print "Parsing $orig_field as $parse_as (time) (value = $value)\n";
					if (!($invalid_ret = is_invalid_time($value))) {
						$parsing_as = "time";
						$val = $value;
					} else {
						if ($invalid_ret == 4){
							$was_whitespace = 1;
						}
					}
				}
			};

			if (!$parsing_as || $@ || $invalid_ret) {
				if ($invalid_ret && $was_whitespace){
					report('field_contained_whitespace', line => $line_number, value => $value, field => $orig_field);
				} else {
					report('field_failed_to_parse', line => $line_number, value => $value, field => $orig_field, parse => $parse_as);
				}
			} elsif (defined($val)) {
				if ((defined($lbound) and $lbound ne "" and $val < $lbound) or (defined($ubound) and $ubound ne "" and $val > $ubound)) {
					if (!defined($lbound) or $lbound eq "") {
						$lbound = "-inf";
					} elsif (!defined($ubound) or $ubound eq "") {
						$ubound = "inf";
					}
					if ($warning) {
						warning($warning, line => $line_number, field => $orig_field, value => $value, unit => $unit, parse_as => $parse_as, lbound => $lbound, ubound => $ubound);
					} elsif ($error) {
						error($error, line => $line_number, field => $orig_field, value => $value, unit => $unit, parse_as => $parse_as, lbound => $lbound, ubound => $ubound);
					} else {
						report('field_out_of_bounds', line => $line_number, field => $orig_field, value => $value, unit => $unit, parse_as => $parse_as, lbound => $lbound, ubound => $ubound);
					}
				} ## end if ((defined($lbound) ...
			} ## end elsif (defined($val))
		} ## end foreach my $field (@fields)

		if (defined($data_line{"lon"}) and defined($data_line{"lat"}) and !is_invalid_float($data_line{"lon"}) and !is_invalid_float($data_line{"lat"}) and not is_in_water($data_line{"lat"}, $data_line{"lon"})) {
#			print $data_line{"lon"} . ",", $data_line{"lon"} . " | " . is_invalid_float($data_line{"lon"}) . " | " . is_invalid_float($data_line{"lat"}) . "\n";
			report('data_bathymetry_failed', line => $line_number, method => uc($bathymetry_name), longitude => $data_line{"lon"}, latitude => $data_line{"lat"});
		}

		if (defined($data_line{"date"})) {
			my $arr = $all_fields->{"date"};
			my (undef, undef, $lbound, $ubound, undef, undef) = @$arr;
			if ($ubound eq "today") {
				$ubound = get_today();
			}
			my $date = $data_line{"date"};
			my $invalid = is_invalid_date($date, $lbound, $ubound);
			if ($invalid == 1) {
				report('data_invalid_date', line => $line_number, value => $date);
			} elsif ($invalid == 2) {
				report('data_date_bounds_error', line => $line_number, value => $date, lbound => $lbound, ubound => $ubound);
			} elsif ($invalid == 3) {
				report('data_invalid_month', line => $line_number, value => $date);
			} elsif ($invalid == 4) {
				report('data_invalid_day_of_month', line => $line_number, value => $date);
			} elsif ($invalid == 5) {
				report('data_pre_1975_detected', line => $line_number, value => $date);
			}

		} ## end if (defined($data_line...

		if (defined($data_line{"year"})) {
			my (undef, undef, undef, $dayOfMonth, $month, $year, undef, $julian_day, undef) = gmtime();
            $julian_day += 1;
            $month += 1;
			$year       += 1900;
            my $error = 0;
            eval {
                for my $date_part (qw(year month day julian)){
                    if (defined $data_line{$date_part} && is_invalid_int($data_line{$date_part})){
                        $error = 1;
                    }
                }
            };
            if ($@){
                $error = 1;
            }
			if (!$error && defined $data_line{"month"} and defined $data_line{"day"}) {
				my $arr = $all_fields->{"day"};
				my (undef, undef, $day_lbound, undef, undef, undef) = @$arr;
				$arr = $all_fields->{"year"};
				my (undef, undef, $year_lbound, undef, undef, undef) = @$arr;
				$arr = $all_fields->{"month"};
				my (undef, undef, $month_lbound, undef, undef, undef) = @$arr;

				my $new_ubound = sprintf("%04i%02i%02i", $year,              $month,              $dayOfMonth);
				my $new_lbound = sprintf("%04i%02i%02i", $year_lbound,       $month_lbound,       $day_lbound);
				my $date       = sprintf("%04i%02i%02i", $data_line{"year"}, $data_line{"month"}, $data_line{"day"});
				my $invalid    = is_invalid_date($date,  $new_lbound,        $new_ubound);

				if ($invalid == 1) {
					report('data_invalid_date', line => $line_number, value => $date);
				} elsif ($invalid == 2) {
					report('data_date_bounds_error', line => $line_number, value => $date, lbound => $new_lbound, ubound => $new_ubound);
				} elsif ($invalid == 3) {
					report('data_invalid_month', line => $line_number, value => $date);
				} elsif ($invalid == 4) {
					report('data_invalid_day_of_month', line => $line_number, value => $date);
				} elsif ($invalid == 5) {
					report('data_pre_1975_detected', line => $line_number, value => $date);
				}
			} ## end if (defined $data_line...
			if (!$error && defined($data_line{"jd"})) {
				my $arr = $all_fields->{"jd"};
				my (undef, undef, $jd_lbound, undef, undef, undef) = @$arr;
				$arr = $all_fields->{"year"};
				my (undef, undef, $year_lbound, undef, undef, undef) = @$arr;
				my $new_lbound = sprintf("%04i%03i",    $year_lbound,       $jd_lbound);
				my $new_ubound = sprintf("%04i%03i",    $year,              $julian_day);
				my $date       = sprintf("%04i%03i",    $data_line{"year"}, $data_line{"jd"});
				my $invalid    = is_invalid_date($date, $new_lbound,        $new_ubound);

				if ($invalid == 1) {
					report('data_invalid_date', line => $line_number, value => $date);
				} elsif ($invalid == 2) {
					report('data_date_bounds_error', line => $line_number, value => $date, lbound => $new_lbound, ubound => $new_ubound);
				} elsif ($invalid == 5) {
					report('data_pre_1975_detected', line => $line_number, value => $date);
				} elsif ($invalid == 6) {
					report('data_invalid_julian', line => $line_number, value => $date);
				}
			} ## end if (defined($data_line...

		} ## end if (defined($data_line...
	} ## end for (my $i = $data_begin...
} ## end sub check_data

#--------------------------------------------------------------------------------------------------
# make_data_hash($line_number, $line)
#   Makes a hash out of the data line, with (field_name => value) pairs
#
# Reports:
#	If a line doesn't contain the delimiter from the header: _bad_data_line_delimiter
#--------------------------------------------------------------------------------------------------
sub make_data_hash {
	my $number = shift;
	my $line   = lc(shift);
	my @split;
	if (defined($headers{'/delimiter'}) && $headers{'/delimiter'} && defined($data_delimiter)) {
		if ($line =~ /$data_delimiter/) {
			@split = split(/$data_delimiter/, $line);
		} else {
			report('bad_data_line_delimiter', line => $number, delim => $headers{'/delimiter'});
			@split = split(/[,;]|\s+/, $line);
		}
	} else {
		@split = split(/[,;]|\s+/, $line);
	}
	my %ret;
	
	if ($#fields != $#split){
	    return %ret;
	}
    
    my $fake_field_counter = 0;
	foreach my $field (@fields) {
		my $val = ($config{general}{strict_delim} ? shift(@split) : trim(shift(@split)));
		if (!defined($val)) {
			last;
		} elsif (exists($ret{$field})){
		    $ret{"fake_field_$fake_field_counter"} = ++$fake_field_counter;
		} else {
			if (is_missing($val)) {
				$ret{$field} = undef;
			} else {
				$ret{$field} = $val;
			}
		} ## end else [ if (!defined($val))
	} ## end foreach my $field (@fields)

	return %ret;
} ## end sub make_data_hash

#--------------------------------------------------------------------------------------------------
# check_fields()
#	Checks the fields for correct units, duplicate fields, unknown fields
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#   read_header has been called
#
# Reports:
#	There are more or less fields than units: _unbalanced_units_and_fields
#	Field is listed with incorrect unit: _field_has_wrong_unit
#	A field is repeated: _duplicate_fields_found
#   A field isn't recognized: _fields_not_recognized
#   A field with a suffix doesn't have an associated field without one: _suffix_doesnt_match
#--------------------------------------------------------------------------------------------------
sub check_fields {
	if (not defined($headers{'/fields'}) or not defined($headers{'/units'})) {
		return;
	}
	my @fields = split(/,/, $headers{'/fields'});
	my @units  = split(/,/, $headers{'/units'});

	my $max_fields = min($#fields, $#units);
	if ($#fields != $#units) {
		report('unbalanced_units_and_fields', header => '/fields', numunits => scalar(@units), numfields => scalar(@fields));
	}

	my (%seen, @dupes, @fields_not_found);
	for (my $i = 0; $i <= $max_fields; $i += 1) {
		my $orig_field = lc($fields[$i]);
		my $unit  = $units[$i];
		if ($seen{$orig_field}++) {
			if ($seen{$orig_field} == 2) {
				push(@dupes, $orig_field);
			}
		}

		my $unsuffixed_base = $orig_field;

#		my ($has_suffix, $keep_checking_suffix) = (0, 1);
#		while ($keep_checking_suffix){
#			$keep_checking_suffix = 0;
#			foreach my $suffix (@{$config{'suffixes'}}){
#				if ($field_base =~ /_(?:\d*\.?\d*)?$suffix$/ && !defined($config{'fields'}{$field})){
#					$has_suffix = $keep_checking_suffix = $suffix;
#					$field_base =~ s/_(?:\d*\.?\d*)?$suffix$//;
#					last;
#				}
#			}
#		}

		my ($suffix_unit, $has_suffix, $keep_checking_dims, $suffix_needs_full_field) = ('', 0, 1, 1);
		keys(%{$config{'suffixes'}});
		while (my ($suffix, $info) = each(%{$config{'suffixes'}})){
			if ($unsuffixed_base =~ /^$suffix|$suffix$/){
				$has_suffix = $suffix;
				# $unsuffixed_base =~ s/_$suffix$//; # changed to the below line to take prefixes as well as suffixes
				$unsuffixed_base =~ s/^$suffix|$suffix$//g;
				$suffix_unit = $info->[0];
				$suffix_needs_full_field = $info->[1];
				# print "Line 1149: " . Dumper($has_suffix ); 
				last;
			}
		}

		my $field_base = $unsuffixed_base;

		my $d_re = qr/(?:\d+\.?\d*)/;
		while ($keep_checking_dims){
			$keep_checking_dims = 0;
			foreach my $suffix (@{$config{'nd_fields'}}){
				my $re;
				if ($suffix->[1] == 0){
					$re = qr/_$suffix->[0]$/;
				} elsif ($suffix->[1] == 1){
					$re = qr/_$d_re$suffix->[0]$/;
				} elsif ($suffix->[1] == 2){
					$re = qr/_$suffix->[0]$d_re$/;
				}
				if ($field_base =~ $re){
					$keep_checking_dims = 1;
					$field_base =~ s/$re//;
				}
			}
		}

		$field_base =~ s/(?<![-_])\d{3,5}(?:\.\d+)?$//;    #take out wavelength, the ###.# at the end of the field name

		if (defined($config{'fields'}{$field_base})) {
			my $matched = 0;
			my @valid_units;

			if ($has_suffix){
				my $found_base = 0;
				for (0 .. $max_fields){
					if (lc($fields[$_]) eq $unsuffixed_base){
						$found_base = 1;
						if ($unit eq $units[$_]){
							$matched = 1;
						} else {
							push(@valid_units, $units[$_]);
						}
						last;
					}
				}
				
				if ($suffix_unit){
					if ($unit eq $suffix_unit){
						$matched = 1;
					} else {
						$matched = 0;
						@valid_units = ($suffix_unit);
					}
				}

				if (!$found_base){
					my $error = 1;
					if (!$suffix_needs_full_field){
						for my $field (@fields){
							if ($field =~ /^$unsuffixed_base$WL/i){
								$error = 0;
								last;
							}
						}
					}
					if ($error){
						report('suffix_doesnt_match', header => "/fields", field => $orig_field, base_field => $unsuffixed_base);
					}
                    $matched = 1;
				}
			} else {
                @valid_units = split(/\s*,\s*/, $config{'fields'}{$field_base}[0]);
				foreach my $valid_unit (@valid_units) {
					if ($unit eq $valid_unit) {
						$matched = 1;
						last;
					}
				} ## end foreach my $valid_unit (@valid_units)
			}
			if (!$matched) {
                my $units_flat = join(",", @valid_units);
                if ($units_flat =~ /^(.*?),([^,]*?)$/) {
                    if (@valid_units > 2) {
                        $units_flat = "$1, or $2";
                    } else {
                        $units_flat = "$1 or $2";
                    }
                } ## end if ($units_flat =~ /^(.*?),([^,]*?)$/)
                report('field_has_wrong_unit', field => $orig_field, unit => $units_flat, bad_unit => $unit);
            } ## end if (not $matched)
		} else {
			push(@fields_not_found, $orig_field);
		}
	} ## end for (my $i = 0; $i <= $max_fields...

	if (@dupes) {
		report('duplicate_fields_found', header => "/fields", fields => join(' ', @dupes));
	}

	if (@fields_not_found) {
		my $fields = "[" . join(' ', @fields_not_found) . "] " . ($#fields_not_found > 0 ? "are" : "is");
		report('fields_not_recognized', header => "/fields", fields => $fields);
	}

} ## end sub check_fields

#--------------------------------------------------------------------------------------------------
# check_validity_section()
#	Checks the compares set in the [validity] section of the config file.
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#   read_header has been called
#
# Reports:
#   Warnings or errors set in the [validity] section
#   A value cannot be parsed as listed: _validity_failed_to_parse
#--------------------------------------------------------------------------------------------------

sub check_validity_section {
	my $validity_config = $config{'validity'};
	foreach my $validity_ref (@$validity_config) {
		my @row_array = @$validity_ref;
		my ($header, $validity, $modifiers, $values, $warning, $error) = @row_array;
		if (not $warning and not $error) {
			$error = "validity_failed";
		}
		if (not $modifiers) {
			$modifiers = "equals exact";
		} else {
			if ($modifiers !~ /equals|contains/i) {
				$modifiers = "equals $modifiers";
			} elsif ($modifiers !~ /exact|any|all/i) {
				$modifiers .= " exact";
			}
		} ## end else [ if (not $modifiers)
		$modifiers = " $modifiers ";

		my @values = ();
		my $delim  = ",";

		

		if ($modifiers =~ /exact/i) {
			push(@values, $values);
		} elsif ($modifiers =~ /any(.*?)\s/i) {
			if ($1) {
				$delim = $1;
			}
			push(@values, split(/\s*$delim\s*/, $values));
		} elsif ($modifiers =~ /all(.*?)\s/i) {
			if ($1) {
				$delim = $1;
			}
			push(@values, split(/\s*$delim\s*/, $values));
		} ## end elsif ($modifiers =~ /all(.*?)\s/i)

		foreach my $header (get_matching_headers($header)) {
			my ($symbol, $parse_as);
			if ($modifiers =~ /(float|int)/) {
				$parse_as = $1;
				if ($validity eq "valid") {
					$symbol = "!=";
				} else {
					$symbol = "==";
				}
			} else {
				if ($validity eq "valid") {
					$symbol = "!~";
				} else {
					$symbol = "=~";
				}
			} ## end else [ if ($modifiers =~ /(float|int)/)

			my $matches = ($modifiers =~ /all/i);

			my $header_value = $headers{$header};
			if (not defined($header_value)) {
				next;
			}

			if ($parse_as) {
				if ($parse_as eq "int") {
					eval {$header_value = sprintf("%d", $header_value);};
				} elsif ($parse_as eq "float") {
					eval {$header_value = sprintf("%0.32g", $header_value);};
				}

				if ($@) {
					next;
				}
			} ## end if ($parse_as)

			foreach my $validity_value (@values) {
				if ($validity_value =~ m"^/" and exists($headers{$validity_value})) {
					if ($validity_value eq "/missing"){
						push(@values, @missing);
					} else {
						$validity_value = $headers{$validity_value};
						if ($validity_value =~ /,/) {
							my @more_values = split(/,/, $validity_value);
							$validity_value = shift(@more_values);
							push(@values, @more_values);
						} 
					}
				} ## end if ($validity_value =~...
				if ($modifiers =~ /contains/) {
					if ($header_value =~ /\Q$validity_value\E/i and $modifiers =~ /any|exact/) {
						$matches = 1;
						last;
					} elsif ($header_value !~ /\Q$validity_value\E/i and $modifiers =~ /all/) {
						$matches = 0;
						last;
					}
				} else {
					if ($parse_as) {
						if ($parse_as =~ /int/) {
							eval {
								if ($validity_value =~ /\./ or $validity_value != int($validity_value)) {
									report('validity_failed_to_parse', header => $header, value => $validity_value, type => $parse_as);
								}
								$validity_value = sprintf("%d", $validity_value);
							};
						} elsif ($parse_as =~ /float/) {
							eval {$validity_value = sprintf("%0.32g", $validity_value);};
						}

						if ($@) {
							report('validity_failed_to_parse', header => $header, value => $validity_value, type => $parse_as);
							next;
						}

						if ($header_value == $validity_value and $modifiers =~ /any|exact/) {
							$matches = 1;
							last;
						} elsif ($header_value != $validity_value and $modifiers =~ /all/) {
							$matches = 0;
							last;
						}
					} else {
						if ($header_value eq $validity_value and $modifiers =~ /any|exact/) {
							$matches = 1;
							last;
						} elsif ($header_value ne $validity_value and $modifiers =~ /all/) {
							$matches = 0;
							last;
						}
					} ## end else [ if ($parse_as)
				} ## end else [ if ($modifiers =~ /contains/)
			} ## end foreach my $validity_value ...

			if (($matches and $validity eq "invalid") or (not $matches and $validity eq "valid")) {
				my $statement = "$header should ";
				if ($validity eq "invalid") {
					$statement .= "not ";
				}
				if ($modifiers =~ /contains/) {
					$statement .= "contain ";
					if ($modifiers =~ /(any|all)/) {
						$statement .= "$1 ";
					}
				} else {
					$statement .= "equal one ";
				}
				$statement .= "of the following: ";

				my $values_flat = join("$delim ", @values);
				if ($values_flat =~ /^(.*?)$delim\s([^$delim]*?)$/) {
					if (@values > 2) {
						$values_flat = "$1, or $2";
					} else {
						$values_flat = "$1 or $2";
					}
				} ## end if ($values_flat =~ /^(.*?)$delim\s([^$delim]*?)$/)
				$statement .= "[$values_flat].";

				if ($warning) {
					warning($warning, header => $header, value => $header_value, statement => $statement);
				} elsif ($error) {
					error($error, header => $header, value => $header_value, statement => $statement);
				}

				if ($modifiers =~ /contains\s+any/) {
					$values_flat = quotemeta(join('', @values));
					$header_value =~ s/[$values_flat]//g;
					$headers{$header} = $header_value;
				}
			} ## end if (($matches and $validity...
		} ## end foreach my $header (get_matching_headers...
	} ## end foreach my $validity_ref (@$validity_config)
} ## end sub check_validity_section

#--------------------------------------------------------------------------------------------------
# check_header_compares()
#	Checks the compares set in the [header_value_comparison_problem] section of the config file.
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#   read_header has been called
#
# Reports:
#   Warnings or errors set in the [header_value_comparison_problem] section
#--------------------------------------------------------------------------------------------------
sub check_header_compares {
	my $compares_config = $config{'header_value_comparison_problem'};
	foreach my $comp_ref (@$compares_config) {
		my @comp = @$comp_ref;
		my ($head1, $symbol, $head2, $warning, $error) = @comp;
		if (not $warning and not $error) {
			$error = "header_value_comparison_problem";
		}
		my $val1 = $headers{$head1};
		my $val2 = $headers{$head2};

		if (not defined($val1) or not defined($val2)) {
			next;
		}

		#drop bracketed units
		if ($val1 =~ /(.*?)\[.*?\]/) {
			$val1 = $1;
		}
		if ($val2 =~ /(.*?)\[.*?\]/) {
			$val2 = $1;
		}
		my $bad_compare;
		if ($symbol =~ /[=<>!]/) {
			eval {
				$val1 = sprintf("%f", $val1);
				$val2 = sprintf("%f", $val2);
			};

			if ($@) {
				next;
			}

			if (!eval("$val1 $symbol $val2")) { ## no critic
				$bad_compare = 1;
			}
		} else {
			if (!eval("'$val1' $symbol '$val2'")) { ## no critic
				$bad_compare = 1;
			}
		}

		if ($bad_compare) {
			my $compare_text = "";
			if ($symbol eq "<>" or $symbol eq "!=") {
				$compare_text = "not be equal to";
			} elsif ($symbol =~ /^([l<])|([g>])/) {
				if ($1) {
					$compare_text = "be less than";
				} elsif ($2) {
					$compare_text = "be greater than";
				}
				if ($symbol =~ /[=e]$/) {
					$compare_text .= " or equal to";
				}
			} elsif ($symbol eq "==") {
				$compare_text = "be equal to";
			}
			if ($error) {
				error($error, value1 => $val1, value2 => $val2, header1 => $head1, header2 => $head2, compare => $compare_text);
			} elsif ($warning) {
				warning($warning, value1 => $val1, value2 => $val2, header1 => $head1, header2 => $head2, compare => $compare_text);
			}
		} ## end if ($bad_compare)
	} ## end foreach my $comp_ref (@$compares_config)

} ## end sub check_header_compares

#--------------------------------------------------------------------------------------------------
# read_header()
#   Populates %headers, checks some errors
#   Lines starting with ! are comments and are ignored and dropped
#
# Pre-conditions:
#   our %errors, %warnings, %headers exists
#   read_config has been called
#   our @lines = slurped input file
#   our $data_begin declared
#
# Post-conditions:
#   %headers contains header lines as (/header => value) pair
#	$data_begin = index of the first line of data in @lines
#
# Reports:
#	Two of the same headers are found: _duplicate_header
#       In this event, only the first is kept
#	An = is found with no value afterwards: _header_without_value_detected
#	No /begin_header tag found: _no_starting_begin_header
#	/end_header@ found instead of /end_header: _at_symbol_in_end_header
#	/end_header isn't found at all: _no_ending_end_header
#	Single comma detected at end of line (joins lines): _line_split_trailing_comma_detected
#	File started with too many non-header lines, assumed not seabass file: _not_seabass_file
#	A line not started with / or ! is found: _invalid_header_line
#	Empty line in data: _empty_line
#	Leading or trailing spaces: _spaces_around_line
#--------------------------------------------------------------------------------------------------
sub read_header {
	my ($started, $bad_starting_lines, $no_ending_header, $i) = (0, 1, 0, 0);
	
	my $last_header_line = first_regex('/end_header',@lines);
	if (!defined($last_header_line)){
		$last_header_line = $#lines;
		$no_ending_header = 1;
	}
	
	for (; $i <= $last_header_line; $i++) {
		my $line = $lines[$i];
		if (!defined($line)){
		    next;
		}
		chomp($line);

		if ($line =~ m"^(\xef\xbb\xbf|\xff\xfe(?:00)?|(?:00)?\xfe\xff)+/begin_header") {
			my $bytes = join(' ', map {sprintf("0x%s", uc(unpack("(H2)*")))} split(//, $1));
			report('bom_detected', bytes => $bytes);
		}

		if ($line =~ /^\s*$/){
			$bad_starting_lines += (!$started); #If the real header hasn't started, blank lines are bad.
			report('empty_line', line => $i+1, block => 'Header');
			next;
		} elsif ($line =~ /^\s+|\s+$/){
			report('spaces_around_line', line => $i+1, block => 'header');
			strip($line);
		}
		if ($line =~ m"^/") {
			while ($line =~ /[^,]+,$/ and ($i + 1) <= $#lines and $lines[$i + 1] !~ m"^/") {
				$i++;
				report('line_split_trailing_comma_detected', line => $i);
				$line .= $lines[$i];
				chomp($line);
			} ## end while ($line =~ /,$/ and ...
		} ## end if ($line =~ m"^/")

		if ($line =~ m"^(/\S+?)(=(.*?))?$") {
			my ($header_name, $header_value_with_equals, $header_value) = (lc($1), $2, $3);
			if ($headers{$header_name}) {
				report('duplicate_header', header => $header_name);
			} else {
				if ($header_value_with_equals) {
					if (defined($config{'headers'}{$header_name}) and (($config{'headers'}{$header_name}[0] & 4) != 0)) {
						report('no_value_header_with_value', header => $header_name, value => $header_value);
						$headers{$header_name} = 1;
					} elsif (length($header_value) != 0) {
						$headers{$header_name} = lc($header_value);
					} else {
						report('header_without_value_detected', header => $header_name);
						$headers{$header_name} = "";
					}
				} elsif (defined($config{'headers'}{$header_name}) and (($config{'headers'}{$header_name}[0] & 4) != 0)) {
					$headers{$header_name} = 1;
				} else {
					report('header_without_value_detected', header => $header_name);
				}
			} ## end else [ if ($headers{$header_name...
			if (not $started and $header_name !~ m"^/begin_header@?") {
				report('no_starting_begin_header');
			} elsif ($header_name =~ m"^/end_header(@?)") {
				if ($1) {
					$headers{"/end_header"} = 1;
				}
				$i++;
				last;
			} ## end elsif ($header_name =~ m"^/end_header(@?)")
			$started = 1;
			$bad_starting_lines = 0;
		} elsif ($line =~ /^!/) {
			next;
		} elsif (not $started) {
			$bad_starting_lines++;
			if ($bad_starting_lines > $MAXIMUM_BAD_STARTING_LINES + 1) {
				$bad_file = 1;
				report('not_seabass_file');
				last;
			}
		} elsif ($no_ending_header){
			last;
		} else {
			report('invalid_header_line',line => ($i+1));
		}
	} ## end for (; $i <= $#lines; $i...
	if ($bad_starting_lines and !$bad_file){
		$bad_file = 1;
		report('not_seabass_file');
	}
	
	if (!$bad_file && $no_ending_header){
		report('no_ending_end_header');
	}
	
	$data_begin = $i;
} ## end sub read_header

#--------------------------------------------------------------------------------------------------
# check_for_invalid_numbers()
#	Checks the settings in the [numbers] section of the config file
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#   read_header has been called
#
# Reports:
#	If a header value is found as a range: _range_detected
#	Date doesn't match YYYYMMDD or YYYYJJJ: _invalid_date
#	Month in date isn't between 1 and 12: _invalid_month
#	Day of month doesn't exist: _invalid_day_of_month
#	A date before 1975 is given: _pre_1975_header_date_detected
#	Time doesn't match HH:MM:SS[GMT]: _invalid_time
#	Hours, minutes, or seconds out of range: _invalid_time_value
#	Degree doesn't match float[DEG]: _invalid_header_location_format
#	Degree isn't integer, float, or scientific notation: _invalid_header_location_value
#	Couldn't parse a given float: _invalid_float
#	Couldn't parse a given integer: _invalid_int
#	Date/time/float/degree/int out of bounds: _number_out_of_bounds_error
#--------------------------------------------------------------------------------------------------
sub check_for_invalid_numbers {
	my $numbers_config = $config{'numbers'};
	my %headers_reported;
	while (my ($header, $array) = each(%$numbers_config)) {
		my ($type, $lbound, $ubound) = @$array;

		foreach my $header (get_matching_headers($header)) {
			if ($headers_reported{$header}){
				next;
			}
			if (exists($headers{$header})) {
				# Unnecessary?
#				if ($type !~ /degree|time/) {
#					my $val = $headers{$header};
#					$val =~ s/\[.*$//;
#					$headers{$header} = $val;
#				}

				my $plural;
				my @vals;
				if ($type =~ /s\b/) {
					$plural = 1;
					@vals = split(',', $headers{$header});
				} else {
					@vals = ($headers{$header});
				}

				foreach my $val (@vals) {
					my $invalid = 0;
					if ($header ne '/missing' && is_missing($val)) {
						if ($type =~ /non_null/) {
							report('cant_be_missing_or_none', header => $header);
							$headers_reported{$header} = 1;
						}
						next;
					} ## end if (is_missing($val))

					#Broken
#					if ($val =~ /[^eE]-/) {
#						report('range_detected', header => $header);
#						$val = $1;
#					}

					if ($type =~ /date/) {
						if ($ubound eq "today") {
							$ubound = get_today();
						}
						$invalid = is_invalid_date($val, $lbound, $ubound);
						if ($invalid == 1) {
							report('invalid_date', header => $header, value => $val);
						} elsif ($invalid == 2) {
							report('number_out_of_bounds_error', header => $header, value => $val, lbound => $lbound, ubound => $ubound);
						} elsif ($invalid == 3) {
							report('invalid_month', header => $header, value => $val);
						} elsif ($invalid == 4) {
							report('invalid_day_of_month', header => $header, value => $val);
						} elsif ($invalid == 5) {
							report('pre_1975_header_date_detected', header => $header, value => $val);
						}
					} elsif ($type =~ /time/) {
						$invalid = is_invalid_header_time($val, $lbound, $ubound);
						if ($invalid == 1) {
							report('invalid_time', header => $header, value => $val);
						} elsif ($invalid == 2) {
							report('number_out_of_bounds_error', header => $header, value => $val, lbound => $lbound, ubound => $ubound);
						} elsif ($invalid == 3) {
							report('invalid_time_value', header => $header, value => $val);
						}
					} elsif ($type =~ /degree/) {
						$invalid = is_invalid_header_degree($val, $lbound, $ubound);
						if ($invalid == 1) {
							report('invalid_header_location_format', header => $header, value => $val);
						} elsif ($invalid == 2) {
							report('number_out_of_bounds_error', header => $header, value => $val, lbound => $lbound, ubound => $ubound);
						} elsif ($invalid == 3) {
							report('invalid_header_location_value', header => $header, value => $val);
						}
					} elsif ($type =~ /float/) {
						$invalid = is_invalid_float($val, $lbound, $ubound);
						if ($invalid == 1) {
							report('invalid_float', header => $header, value => $val);
						} elsif ($invalid == 2) {
							report('number_out_of_bounds_error', header => $header, value => $val, lbound => $lbound, ubound => $ubound);
						}
					} elsif ($type =~ /int(?:eger)?/) {
						$invalid = is_invalid_int($val, $lbound, $ubound);
						if ($invalid == 1) {
							report('invalid_int', header => $header, value => $val);
						} elsif ($invalid == 2) {
							report('number_out_of_bounds_error', header => $header, value => $val, lbound => $lbound, ubound => $ubound);
						}
					} ## end elsif ($type =~ /int(eger)?/)

					if ($invalid){
						$headers_reported{$header} = 1;
					}
				} ## end foreach my $val (@vals)

				#take out [DEG] and [GMT], as they have already been checked to contain them
				if ($type =~ /degree|time/) {
					my $new_val = "";
					foreach my $val (@vals) {
						if ($new_val) {
							$new_val .= ",";
						}
						$val =~ s/\[.*$//;
						$new_val .= $val;
					} ## end foreach my $val (@vals)

					$headers{$header} = $new_val;
				} ## end if ($type =~ /degree|time/)
			} ## end if (exists($headers{$header...
		} ## end foreach my $header (get_matching_headers...
	} ## end while (my ($header, $array...
} ## end sub check_for_invalid_numbers

#--------------------------------------------------------------------------------------------------
# get_matching_headers($header)
#   Returns a list of names of all headers matching the given string.
#   If $header starts with a /, that header will be returned.
#   If $header is *, returns every header.
#   Else, if any header contains $header, it is considered matching
#--------------------------------------------------------------------------------------------------
sub get_matching_headers {
	my $header  = shift;
	my @headers = ();
	if ($header =~ m"^/") {
		push(@headers, $header);
	} elsif ($header eq "*") {
		push(@headers, keys(%headers));
	} else {
		foreach my $key (keys(%headers)) {
			if ($key =~ /$header/i) {
				push(@headers, $key);
			}
		}
	} ## end else [ if ($header =~ m"^/")
	return @headers;
} ## end sub get_matching_headers

#--------------------------------------------------------------------------------------------------
# get_today()
#	Returns today's (current GMT's) date as YYYYMMDD
#--------------------------------------------------------------------------------------------------
sub get_today {
	my (undef, undef, undef, $dayOfMonth, $month, $yearOffset, undef, undef, undef) = gmtime();
	return sprintf("%04i%02i%02i", 1900 + $yearOffset, $month+1, $dayOfMonth);
}

#--------------------------------------------------------------------------------------------------
# get_today_julian()
#	Returns today's (current GMT's) date as YYYYJJJ
#--------------------------------------------------------------------------------------------------
sub get_today_julian {
	my (undef, undef, undef, $dayOfMonth, $month, $yearOffset, undef, $yday, undef) = gmtime();
	return sprintf("%04i%03i", 1900 + $yearOffset, $yday+1);
}

#--------------------------------------------------------------------------------------------------
# get_today_year()
#	Returns today's (current GMT's) year as YYYY
#--------------------------------------------------------------------------------------------------
sub get_today_year {
	my (undef, undef, undef, undef, undef, $yearOffset, undef, undef, undef) = gmtime();
	return sprintf("%04i", 1900 + $yearOffset);
}

#--------------------------------------------------------------------------------------------------
# is_invalid_date($val[, $lbound[, $ubound]])
#	Checks if a date is in a valid format
#
# Returns:
#	0: date is valid
#	1: date doesn't match YYYYMMDD or YYYYJJJ
#	2: date out of bounds
#	3: invalid month
#	4: invalid day of month
#	5: date before 1975
#	6: invalid Julian day
#	7: contains white space
#--------------------------------------------------------------------------------------------------
sub is_invalid_date {
	my ($val, $lbound, $ubound) = @_;
	if (!defined($val)){
		return 1;
	} elsif ($val =~ /\s/){
		return 7;
	} elsif ($val =~ /^(\d{4})(\d{2})(\d{2})$/) {
		if ($1 < 1975) {
			return 5;
		} elsif ((defined($ubound) and $val > $ubound) or (defined($lbound) and $val < $lbound)) {
			return 2;
		}
		my @m = @months;
		if (is_leap_year($1)) {
			@m = @months_leap;
		}
		if ($2 <= 0 or $2 > 12) {
			return 3;
		}
		if ($3 <= 0 or ($3 > $m[$2])) {
			return 4;
		}
		return 0;
	} elsif ($val =~ /^(\d{4})(\d{3})$/) {
		if ($1 < 1975) {
			return 5;
		} elsif ((defined($ubound) and $val > $ubound) or (defined($lbound) and $val < $lbound)) {
			return 2;
		}
		my $add = 0;
		if (is_leap_year($1)) {
			$add = 1;
		}
		if ($2 <= 0 or $2 > (365 + $add)) {
			return 6;
		}
		return 0;
	} ## end elsif ($val =~ /^(\d{4})(\d{3})$/)
	return 1;
} ## end sub is_invalid_date

#--------------------------------------------------------------------------------------------------
# is_leap_year($year)
#	Checks if the given $year is a leapyear
#
# Returns:
#   1: $year is a leapyear
#   undef: %year is NOT a leapyear
#--------------------------------------------------------------------------------------------------
sub is_leap_year {
	my $year = shift;
	if ($year % 400 == 0 || ($year % 100 != 0 && $year % 4 == 0)) {
		return 1;
	}
	return;
} ## end sub is_leap_year

#--------------------------------------------------------------------------------------------------
# is_invalid_header_time($val[, $lbound[, $ubound]])
#	Checks if a time is in a valid format, with the [GMT] label
#
# Returns:
#	0: time is valid
#	1: time doesn't match HH:MM:SS[GMT]
#	2: time out of bounds
#	3: invalid value for hours, minutes, or seconds
#--------------------------------------------------------------------------------------------------
sub is_invalid_header_time {
	my ($val, $lbound, $ubound) = @_;
	if (!defined($val)){
		return 1;
	} elsif ($val =~ /^(\d{1,2}):(\d{2}):(\d{2})(?:\.\d*)?\[GMT\]$/i) {
		if ((defined($ubound) and $ubound ne "" and $val > $ubound) or (defined($lbound) and $lbound ne "" and $val < $lbound)) {
			return 2;
		} elsif ($1 < 0 or $1 > 23 or $2 < 0 or $2 > 59 or $3 < 0 or $3 > 59) {
			return 3;
		}
		return 0;
	} ## end if ($val =~ /^(\d{2}):(\d{2}):(\d{2})\[GMT\]$/i)
	return 1;
} ## end sub is_invalid_header_time

#--------------------------------------------------------------------------------------------------
# is_invalid_time($val[, $lbound[, $ubound]])
#	Checks if a time is in a valid format
#
# Returns:
#	0: time is valid
#	1: time doesn't match HH:MM:SS
#	2: time out of bounds
#	3: invalid value for hours, minutes, or seconds
#	4: contains white space
#--------------------------------------------------------------------------------------------------
sub is_invalid_time {
	my ($val, $lbound, $ubound) = @_;
	if (!defined($val)){
		return 1
	} elsif ($val =~ /\s/){
		return 4;
	} elsif ($val =~ /^(\d{1,2}):(\d{2}):(\d{2}(?:\.\d*)?)$/i) {
		if ((defined($ubound) and $val > $ubound) or (defined($lbound) and $val < $lbound)) {
			return 2;
		} elsif ($1 < 0 or $1 > 23 or $2 < 0 or $2 > 59 or ($3 and ($3 < 0 or $3 >= 60))) {
			return 3;
		}
		return 0;
	} ## end if ($val =~ /^(\d{2}):(\d{2}):(\d{2})$/i)
	return 1;
} ## end sub is_invalid_time

#--------------------------------------------------------------------------------------------------
# is_invalid_header_degree($val[, $lbound[, $ubound]])
#	Checks if a [DEG] labeled degree is in a valid format
#
# Returns:
#	0: degree is valid
#	1: doesn't match format float[DEG]
#	2: degree out of bounds
#	3: degree isn't integer, float, or scientific notation
#--------------------------------------------------------------------------------------------------
sub is_invalid_header_degree {
	my ($val, $lbound, $ubound) = @_;
	if (!defined($val)){
		return 1;
	} elsif ($val =~ /^(.*?)\[DEG\]$/i) {
		eval {$val = sprintf("%0.32g", $1);};
		return 3 if ($@);

		if ((defined($ubound) and $val > $ubound) or (defined($lbound) and $val < $lbound)) {
			return 2;
		}
		return 0;
	} ## end if ($val =~ /^(.*?)\[DEG\]$/i)
	return 1;
} ## end sub is_invalid_header_degree

#--------------------------------------------------------------------------------------------------
# is_invalid_float($val[, $lbound[, $ubound]])
#	Checks if a float is in a valid format (float or scientific notation)
#
# Returns:
#	0: float is valid
#	1: value isn't integer, float, or scientific notation
#	2: float out of bounds
#	3: contains white space
#--------------------------------------------------------------------------------------------------
sub is_invalid_float {
	my ($val, $lbound, $ubound) = @_;
	if (!defined($val)){
		return 1;
	} elsif ($val =~ /\s/){
		return 3;
	} elsif ($val eq "" || $val =~ /[a-df-z]/i) {
		return 1;
	}
	eval {$val = sprintf("%0.32g", $val);};
	return 1 if ($@);

	if ((defined($ubound) and $val > $ubound) or (defined($lbound) and $val < $lbound)) {
		return 2;
	}

	return 0;
} ## end sub is_invalid_float

#--------------------------------------------------------------------------------------------------
# is_invalid_int($val[, $lbound[, $ubound]])
#	Checks if an int is in a valid format
#
# Returns:
#	0: int is valid
#	1: value isn't integer
#	2: int out of bounds
#	3: contains white space
#--------------------------------------------------------------------------------------------------
sub is_invalid_int {
	my ($val, $lbound, $ubound) = @_;
	if (!defined($val)){
		return 1;
	} elsif ($val =~ /\s/){
		return 3;
	} elsif ($val eq "" || $val =~ /[a-df-z]/i) {
		return 1;
	}
	if ($val != int($val)) {
		return 1;
	} elsif ((defined($ubound) and $val > $ubound) or (defined($lbound) and $val < $lbound)) {
		return 2;
	}
	return 0;
} ## end sub is_invalid_int

#--------------------------------------------------------------------------------------------------
# check_headers_for_whitespace()
#	Checks for whitespace within header lines.  Any whitespace is removed and errors are reported.
#
# Pre-conditions:
#   our %config, %errors, and %warnings exists
#	read_header() called
#
# Post-conditions:
#	@header_lines will contain no whitespace
#
# Reports:
#	Whitespace found in header: _headerline_contained_whitespace
#
#--------------------------------------------------------------------------------------------------
sub check_headers_for_whitespace {
	while (my ($key, $value) = each(%headers)) {
		if ("$key$value" =~ /\s+/) {
			report('headerline_contained_whitespace', line => "$key=$value");
			delete $headers{$key};
			$key   =~ s/\s+//g;
			$value =~ s/\s+//g;
			$headers{$key} = $value;
			keys(%headers);
		} ## end if ("$key$value" =~ /\s+/)
	} ## end while (my ($key, $value) ...
} ## end sub check_headers_for_whitespace

#--------------------------------------------------------------------------------------------------
# read_config($filename)
#
# Pre-conditions:
#	our %config exists
#
# Post-conditions:
#	%config contains the parsed contents from $filename
#
#--------------------------------------------------------------------------------------------------
sub read_config {
	my $filename     = shift;
	my @config_lines = slurp($filename);
	my $cur_section  = "";

	my %header_config;
	my @header_compares;
	my %numbers;
	my %strings;
	my %field_config;
	my @validity_config;
	my %report_modifiers;
    my %general_section;
    my %suffixes;
	my @nd_fields;

	foreach (@config_lines) {
		$_ =~ s/^#.*|[^\\]#[^|]*//g;    #remove comments
		strip($_);                      #remove leading/trailing whitespace
		$_ =~ s/\\#/#/g;                #unescape the comment character (#)
		next unless ($_);

		if ($_ =~ /^\[(.*?)\]/) {
			$cur_section = $1;
			next;
		}

		if ($cur_section ne 'strings') {
			$_ = lc($_);
		}

		my @args = split(/\|/, $_);
		strip(@args);
		@args = map { if ($_ eq ""){undef} else {$_}} @args;

		if ($cur_section eq 'headers') {
			my $bit_mask = 0;
			if ($args[1] =~ /required/i) {
				$bit_mask += 1;
			} elsif ($args[1] =~ /optional/i) {

				#$bit_mask += 0;
			} elsif ($args[1] =~ /obsolete/i) {
				$bit_mask += 2;
			}
			if ($args[1] =~ /no_value/i) {
				$bit_mask += 4;
			}
			my $key = shift(@args);
			$args[0] = $bit_mask;
			$header_config{$key} = [@args];
		} elsif ($cur_section eq 'fields') {
			my $key = shift(@args);
			if (!defined($args[1]) || $args[1] !~ /year|month|day|julian|date|time|int(?:eger)?|float|str(?:ing)?/i) {
				$args[1] .= ' float';
			}
			$field_config{$key} = [@args];
		} elsif ($cur_section eq 'header_value_comparison_problem') {
			push(@header_compares, [@args]);
		} elsif ($cur_section eq 'validity') {
			push(@validity_config, [@args]);
		} elsif ($cur_section eq 'numbers') {
			my $key = shift(@args);
			$numbers{$key} = [@args];
		} elsif ($cur_section eq 'strings') {
			my $key = shift(@args);
			$strings{$key} = [@args];
		} elsif ($cur_section eq 'report') {
			my $key = shift(@args);
			$report_modifiers{$key} = [@args];
		} elsif ($cur_section eq 'general') {
			my $key = shift(@args);
			$general_section{$key} = shift(@args);
        } elsif ($cur_section eq 'suffixes') {
            # push(@suffixes, [shift(@args)]);
			my $key = shift(@args);
            # $suffixes{$key} = shift(@args);
            $suffixes{$key} = [@args];
		} elsif ($cur_section eq 'multi_dim_fieldnames') {
			push(@nd_fields, [@args]);
        }
	} ## end foreach (@config_lines)

	$config{'headers'}         = \%header_config;
	$config{'fields'}          = \%field_config;
	$config{'header_value_comparison_problem'} = \@header_compares;
	$config{'validity'}        = \@validity_config;
	$config{'numbers'}         = \%numbers;
	$config{'strings'}         = \%strings;
	$config{'report'}          = \%report_modifiers;
    $config{'general'}         = \%general_section;
    $config{'suffixes'}        = \%suffixes;
    $config{'nd_fields'}       = \@nd_fields;
} ## end sub read_config

#--------------------------------------------------------------------------------------------------
# check_for_required_headers()
#	Checks %headers for required/optional headers
#
# Pre-conditions:
#   our %errors and %warnings exists
#   read_config has been called
#   read_header has been called
#
# Reports:
#   A required header isn't found: _required_header_not_found
#   An optional header isn't found: _optional_header_not_found
#--------------------------------------------------------------------------------------------------
sub check_for_required_headers {
	my $header_config = $config{'headers'};
	while (my ($header, $header_vals) = each(%$header_config)) {
		my ($config_flags, $warning, $error) = @$header_vals;
		my $obsolete = $config_flags & 2;
		if ((not exists($headers{$header}) and not $obsolete) or (exists($headers{$header}) and $obsolete)) {
			if ($warning) {
				warning($warning, header => $header);
			} elsif ($error) {
				error($error, header => $header);
			} else {
				if (not $obsolete) {
					my $required = $config_flags & 1;
					if ($required == 1) {
						report('required_header_not_found', header => $header);
					} elsif ($required == 0) {
						report('optional_header_not_found', header => $header);
					}
				} ## end if (not $obsolete)
			} ## end else [ if ($warning)
		} ## end if ((not exists($headers...
	} ## end while (my ($header, $header_vals...
} ## end sub check_for_required_headers

#--------------------------------------------------------------------------------------------------
# check_for_unknown_headers()
#	Checks %headers for unknown headers
#
# Pre-conditions:
#	our %errors and %warnings exists
#	read_config has been called
#	read_header has been called
#
# Reports:
#	An unrecognized header is found: _unknown_headers
#--------------------------------------------------------------------------------------------------
sub check_for_unknown_headers {
	foreach my $header (keys %headers) {
		if (not exists($config{"headers"}{$header})) {
			report('unknown_header', header => $header);
		}
	}
} ## end sub check_for_unknown_headers

#--------------------------------------------------------------------------------------------------
# report(@strings)
#	Adds the given strings to the error/warning list depending on default severity
#
# Pre-conditions:
#   read_config called
#   our %warnings exists
#	our %errors exists
#
# Post-conditions:
#	%errors or %warnings modified
#--------------------------------------------------------------------------------------------------
sub report {
	my $error_name = shift;
	my $sev        = 2;
	if (exists($config{"strings"}{$error_name})) {
		$sev = @{$config{"strings"}{$error_name}}[0];
	}
	my $string = make_string($error_name, @_);
	if ($sev == 2) {
		push_to_hash(\%errors, $error_name, $string);
		$all_errors{$error_name} ||= {};
		$all_errors{$error_name}{$input_filename}++;
	} elsif ($sev == 1) {
		push_to_hash(\%warnings, $error_name, $string);
		$all_warnings{$error_name} ||= {};
		$all_warnings{$error_name}{$input_filename}++;
	}
} ## end sub report

sub error {
	my $error_name = shift;
	my $string = make_string($error_name, @_);
	push_to_hash(\%errors, $error_name, $string);
	$all_errors{$error_name} ||= {};
	$all_errors{$error_name}{$input_filename}++;
}

sub warning {
	my $error_name = shift;
	my $string = make_string($error_name, @_);
	push_to_hash(\%warnings, $error_name, $string);
	$all_warnings{$error_name} ||= {};
	$all_warnings{$error_name}{$input_filename}++;
}


#--------------------------------------------------------------------------------------------------
# push_to_hash(@strings)
#   Either creates an arrayref in the hash or appends to it.
#--------------------------------------------------------------------------------------------------
sub push_to_hash {
	my ($hash, $name, $item) = @_;

	if (defined($hash->{$name})){
		push(@{$hash->{$name}}, $item);
	} else {
		$hash->{$name} = [$item];
	}
}

#--------------------------------------------------------------------------------------------------
# make_string($string_name, $key1 => $val1, $key2 => $val2, ...)
# make_string($string_name, $key1, $val1, $key2, $val2, ...)
#	Creates a string based on the string or a string in the config file by replacing given fields
#
# Pre-conditions:
#	read_config called
#
# Returns:
#	A string
#--------------------------------------------------------------------------------------------------
sub make_string {
	my $str = shift;
	if (exists($config{"strings"}{$str})) {
		$str = @{$config{"strings"}{$str}}[1];
	}
	my @h = @_;
	while ($#h > 0) {
		my $old = shift(@h);
		my $new = shift(@h);
		if (!defined($new)) {
			$new = 'undef';
		}
		$str =~ s/\{$old\}/$new/g;
	} ## end while ($#h > 0)
	return $str;
} ## end sub make_string

#--------------------------------------------------------------------------------------------------
# slurp($filename)
#   Reads an entire file into an array of strings.
#   If the file fails to open, returns an empty list.
#--------------------------------------------------------------------------------------------------
sub slurp {
	my ($file) = @_;
	open(my $fh, "<", $file) || die "Couldn't slurp $file: $!";
	binmode($fh);
	my @r = <$fh>;
	close($fh);
	return @r;
} ## end sub slurp

#--------------------------------------------------------------------------------------------------
# is_in_water($lat, $lon)
#   Checks the depth (based on the dataset in the config file) and evaluates to true if
#		the location is in water, false otherwise
#	If no bathymetry method was set in config file, merely returns true.
#--------------------------------------------------------------------------------------------------
sub is_in_water {
	if ($bathymetry_name) {
		if ($bathymetry_name eq "getasse30") {
			return $bathymetry_function->(@_, $bathymetry_location) < $getasse30_sea_height;
		} else {    #if ($bathymetry_name =~ /etopo|srtm30|globe/){
			return $bathymetry_function->(@_, $bathymetry_location) < 0;
		}
	} ## end if ($bathymetry_name)
	return 1;
} ## end sub is_in_water

#--------------------------------------------------------------------------------------------------
# max(@list)
#	Returns the maximum value in the list
#--------------------------------------------------------------------------------------------------
sub max {
	my ($max, @vars) = @_;
	for (@vars) {
		$max = $_ if $_ > $max;
	}
	return $max;
} ## end sub max

#--------------------------------------------------------------------------------------------------
# min(@list)
#	Returns the minimum value in the list
#--------------------------------------------------------------------------------------------------
sub min {
	my ($min, @vars) = @_;
	for (@vars) {
		$min = $_ if $_ < $min;
	}
	return $min;
} ## end sub min

#--------------------------------------------------------------------------------------------------
# wrap($text[, $width=90, $delimiter="\n"])
#	Wraps the text to the given width, inserting the delimiter.
#	Additional lines are indented.  If a line starts with whitespace, lines following it will
#		be indented four more spaces than it until another line starting with whitespace is found.
#	The first line is assumed to be a #) started line, so is never indented, but, for the rule
#		above, is considered to be indented once.
#--------------------------------------------------------------------------------------------------
sub wrap {
	my ($text, $width, $delimiter) = @_;
	$width     ||= 90;
	$delimiter ||= "\n";
	my $ret = "";

	my $leading_whitespace = "";

	my @lines = split(/\n/, $text);

	for (my $i = 0; $i <= $#lines; $i++) {
		my $line               = $lines[$i];
		my $leading_whitespace = "";
		if ($line =~ /^(\s+)/) {
			$leading_whitespace = $1;
		}
		while ($line =~ /(?:\G([^\n]{0,$width})(?:\s+|\n|\z))|\G([^\n]{0,$width}\b)|\G([^\n]+?\b)/xmgc) {
			my $string = $1 || $2 || $3 || "";
			if ($string) {
				if ($string =~ /^(\s+)/) {
					$leading_whitespace = "$1    ";
					$ret .= "$string$delimiter";
				} else {
					$ret .= "$leading_whitespace$string$delimiter";
				}
				$leading_whitespace ||= "        ";
			} ## end if ($string)
			if (pos($line) == length($line)) {    #failsafe for Perl < v5.10, otherwise infinite loops
				last;
			}
		} ## end while ($line =~ /(?:\G([^\n]{0,$width})(?:\s+|\n|\z))|\G([^\n]{0,$width}\b)/xmgc)
	} ## end for (my $i = 0; $i <= $#lines...

	return $ret;
} ## end sub wrap

#--------------------------------------------------------------------------------------------------
# strip($var,...)
#	Strips the list of leading and trailing whitespace. Returns an array of all the values.
#	All changes are made in place.
#--------------------------------------------------------------------------------------------------
sub strip {
    s/^\s+|\s+$//g for @_;
    return @_;
}

#--------------------------------------------------------------------------------------------------
# basename($filepath)
#	Gets the file name from the given path.
#--------------------------------------------------------------------------------------------------
sub basename {
	my $file = shift;
	$file =~ s"[^/]*/""g;
	return $file;
}

#--------------------------------------------------------------------------------------------------
# trim($string)
#	Trims and returns the string
#--------------------------------------------------------------------------------------------------
sub trim {
	my $string = shift;
	$string =~ s/^\s+|\s+$//g;
	return $string;
}

#--------------------------------------------------------------------------------------------------
# remove_whitespace_lines_at_end()
#	Checks the end of the file for any blank lines and removes them.
#--------------------------------------------------------------------------------------------------
sub remove_whitespace_lines_at_end {
	while (@lines and !trim($lines[-1])){
		splice(@lines, -1, 1);
	}
}

#--------------------------------------------------------------------------------------------------
# first($needle, @haystack)
#	Finds the first occurance of $needle in $haystack and returns the index, or undef if not found.
#	This uses a string comparison, eq $needle.
#--------------------------------------------------------------------------------------------------
sub first {
	my ($needle, @haystack) = @_;
	my $index = 0;
	while ($index <= $#haystack){
		if ($haystack[$index] eq $needle){
			return $index;
		}
		$index++;
	}
	return;
}

#--------------------------------------------------------------------------------------------------
# first_regex($needle, @haystack)
#	Finds the first occurance of $needle in $haystack and returns the index, or undef if not found.
#	This uses a regex comparison, =~ /$needle/.
#--------------------------------------------------------------------------------------------------
sub first_regex {
	my ($needle, @haystack) = @_;
	my $index = 0;
	while ($index <= $#haystack){
		if ($haystack[$index] =~ /$needle/){
			return $index;
		}
		$index++;
	}
	return;
}


#--------------------------------------------------------------------------------------------------
# dirname($path)
#   Returns the directory part of $path.
#--------------------------------------------------------------------------------------------------
sub dirname {
    my $file = shift;
    $file =~ s"(^.*?)/?[^/]*$"$1";
    return $file || ".";
}

###################################################################################################
#   This is where any non-config-file driven error/warning/sanity checks go.  This is called
#   just before the report is generated. The important variables are as follows:
#
#        %headers, as (/header => value)
#        %errors, as (error_id => [errors])
#        %warnings, as (warning_id => [errors])
#        $all_fields, a hash reference, containing every field known by SeaBASS
#           The hash contains (field_name => [unit,parse_as,lbound,ubound,warning,error]) pairs
#        @fields, containing the names and order of the fields in the data
#        @missing, containing all the /missing values
#        $input_filename, the full or relative path to the input file
#        $data_delimiter, data checks should be safe to say split($data_delimiter, $data_line)
#        @lines, an array containing every line in the input file.
#
#   To iterate through the data
#       make_data_hash($line), makes a hash out of the data line, with (field_name => value) pairs
#
#       for (my $i = $data_begin ; $i <= $#lines ;) {
#           my $line = $lines[$i++];
#           chomp($line);
#           my %data_line = make_data_hash($line);
#           ...
#       }
#
#   To report an error/warning, the functions error, report, and warning are available.
#       report will add an error/warning to the reports according to their severity
#           in the config file
#       error will force the message into the error section
#       warning will force the message into the warning section
#
#       To call these, either use function($error_message) or use the syntax of make_string:
#           function($error_id, name1 => $value1, name2 => $value2, ...)
#           where $error_id is the first field in the [strings] config section and name/value
#               pairs are string replacements
#       If this is confusing, this file and fcheck.pl have plenty of examples, just Ctrl+F report
#
#
#   Any code appended here should be written as if use strict and use warnings were in place.
#   Code should also be commented.  At the very least, add a separater and description of what
#       the subsequent chunk of code is looking for.
#
###################################################################################################
sub one_offs {
    ###################################################################################################
    # Check if the measurement depth is in the header, fields, neither or both.
    #
    my $depthinfields = grep($_ eq 'depth' || $_ eq 'pressure', @fields);
    my $depthinheader = (defined($headers{'/measurement_depth'}) && !is_missing($headers{'/measurement_depth'}));
    
    if (defined($headers{'/measurement_depth'}) and defined($headers{'/data_type'})) {
        if ($headers{'/data_type'} =~ /(sunphoto|above_water)/i && !is_missing($headers{'/measurement_depth'}) && !is_invalid_float($headers{'/measurement_depth'}) && $headers{'/measurement_depth'} != 0) {
            report('depth_should_be_zero', data_type => $1);
        }
    }
    if ($depthinheader && $depthinfields) {
        report('depth_in_header_and_fields');
    } elsif (!$depthinheader && !$depthinfields) {
        report('no_depth_found');
    }
    #
    #################################################
    
    ###################################################################################################
    # Check the locations in header to see if they're on land
    #
    if (defined($headers{'/north_latitude'}) && defined($headers{'/east_longitude'})) {
	my $valid = 1;
	foreach my $h (@headers{qw(/north_latitude /south_latitude /east_longitude /west_longitude)}){
	    if (!defined($h) || is_invalid_float($h)){
		$valid = 0;
		last;
	    }
	}
        if ($valid && (defined($headers{'/data_type'}) && $headers{'/data_type'} =~ /(cast|scan|mooring)/i || ($headers{'/north_latitude'} == $headers{'/south_latitude'} && $headers{'/east_longitude'} == $headers{'/west_longitude'}))) {
            if (not is_in_water($headers{'/north_latitude'}, $headers{'/east_longitude'})) {
                report('bathymetry_failed', latitude => $headers{'/north_latitude'}, longitude => $headers{'/east_longitude'}, method => uc($bathymetry_name));
            }
        }
    }
    #
    #################################################
    
    
    ###################################################################################################
    # Check the date-times in header to make sure it starts before it ends
    #
    if (defined($headers{'/start_date'}) && defined($headers{'/start_time'}) && defined($headers{'/end_date'}) && defined($headers{'/end_time'})) {
        if ("${headers{'/start_date'}}/${headers{'/start_time'}}" gt "${headers{'/end_date'}}/${headers{'/end_time'}}"){
            report('header_start_after_finish', start => "${headers{'/start_date'}}/${headers{'/start_time'}}", end => "${headers{'/end_date'}}/${headers{'/end_time'}}");
        }
    }
    #
    #################################################


    ###################################################################################################
    # Check the number of data rows matches (optional) header
    #
    if (defined($headers{'/number_of_data_rows'})){
		my $number_of_data_rows = $headers{'/number_of_data_rows'};
		my $actual = @lines - $data_begin;
		unless ($actual == $number_of_data_rows){
            report('number_of_data_rows_incorrect', claimed => $number_of_data_rows, actual => $actual);
		}
    }
    #
    #################################################
    
    
    ###################################################################################################
    # Check if the (optional) data_use_warning values match our list of keywords
    #
    if (defined($headers{'/data_use_warning'})) {
    
        my @DATA_USE_WARNINGS_KEYWORDS = ("optically_shallow","experimental","negative_depths","negative_data_values","missing_required_metadata_depth","missing_required_metadata_time");
        my @data_use_warnings = split(',', $headers{'/data_use_warning'});     
	my @invalid_header_values;    	

        foreach my $warning (@data_use_warnings) {
		$warning = lc($warning);
     		unless ( grep(/^$warning$/, @DATA_USE_WARNINGS_KEYWORDS) ) { 	   	        	
			push(@invalid_header_values, $warning);
		}
    	}

	my $msg;
	if (scalar @invalid_header_values >= 1) {
		
		$msg = "\"@invalid_header_values\" either misspelled or not yet in SeaBASS's data_use_warning lexicon";
		report('validity_failed', header => '/data_use_warning', value => $headers{'/data_use_warning'}, statement=> $msg);
	
	}

    }
    #
    #################################################
	

    ###################################################################################################
    # Check if the delimiter is not the preferred comma 
    #
	if (defined($data_delimiter) && $data_delimiter ne ',') {
		report('comma_separator_preferred');
	}
    #
    #################################################
    
    
    ###################################################################################################
    # Check if the file does not end in .sb extension create warning
    #
	if (defined($input_filename) && $input_filename !~ /\.(sb|env)$/) {
		report('file_extension_not_sb', filename => $input_filename);
    }
    #
    #################################################
    
    
    ###################################################################################################
    # Check if the file does not end in .sb extension create warning
    #
	my $assoc_files_no_archives = (defined($headers{'/associated_files'}) && !defined($headers{'/associated_archives'}));
	if ($assoc_files_no_archives){
		report('a_files_present_archives_missing');
	}

	my $assoc_files_no_types = (defined($headers{'/associated_files'}) && !defined($headers{'/associated_file_types'}));
	if ($assoc_files_no_types){
		report('a_files_present_type_missing');
	}

	my $assoc_archives_no_types = (defined($headers{'/associated_archives'}) && !defined($headers{'/associated_archive_types'}));
	if ($assoc_archives_no_types){
		report('a_archives_present_types_missing');
	}
	my @e_files;
	my @e_file_types;
	my @e_archives;
	my @e_archives_types;
	@e_files = split(",", $headers{'/associated_files'} || '');
	@e_file_types = split(",", $headers{'/associated_file_types'} || '');

	@e_archives = split(",", $headers{'/associated_archives'} || '');
	@e_archives_types = split(",", $headers{'/associated_archive_types'} || '');

	# Check for duplicates in associated_files header
	my %e_files_seen;
	foreach my $e_files_idx (@e_files) {
		next unless $e_files_seen{$e_files_idx}++;
		report('duplicate_associated_files');
	}
	# Check that header item counts match.
	if (@e_files != @e_file_types) {
		report('assoc_file_types_count_mismatch');
	}
	# Check for duplicates in associated_archives header
	my %e_archives_seen;
	foreach my $e_archives_idx (@e_archives) {
		next unless $e_archives_seen{$e_archives_idx}++;
		report('duplicate_associated_archives');
	}
	# Check that header item counts match.
	if (@e_archives != @e_archives_types) {
		report('assoc_archives_types_count_mismatch');
	}

	foreach my $e_archives_idx (@e_archives) {
    unless ($e_archives_idx =~ /_associated\.tgz$/) {
        # Code to execute when the element ends with "_associated.tgz"
		report('a_archives_not_associated_tgz', bundle_name => $e_archives_idx);
    } 
	}
    #
    #################################################
    
    
    ###################################################################################################
    # Check if required field is missing frmo env files 
    #
	if($env_arg == 1 && $ini_file ne "$dirname/fcheck.ini"){
		my @fields = split(/,/, $headers{'/fields'});

		##################################
		# add other required fields here #
		##################################
		my @required_env_fields = ('flags', 'lat', 'lon', 'date', 'time');
		# my @required_env_fields = ('flags');

		#join all fields into an array for easy grep
		my $all_fields = join(",", @fields) . "\n";

		# Skip 'date' field if 'year', 'month', and 'day' are present in @fields
		if ($all_fields =~ /year/ && $all_fields =~ /month/ && $all_fields =~ /day/) {
			@required_env_fields = grep { $_ ne 'date' } @required_env_fields;
		}
		# Skip 'time' field if 'hour', 'minute', and 'second' are present in @fields
		if ($all_fields =~ /hour/ && $all_fields =~ /minute/ && $all_fields =~ /second/) {
			@required_env_fields = grep { $_ ne 'time' } @required_env_fields;
		}
		
		my @absent_fields_list; 
		foreach my $field (@required_env_fields) {
			unless (grep { $_ eq $field } @fields) {
				push @absent_fields_list, $field;  # Add missing field to the array
			}
		}
		if (@absent_fields_list) {
			foreach my $absent_fields (@absent_fields_list) {
				# Code to execute when one of the required fields within "@absent_fields_list" is missing
				report('required_field_absent', absent_field => $absent_fields);
			}
		} 
	}
    #
    #################################################
    
    
    ###################################################################################################
}


1;
