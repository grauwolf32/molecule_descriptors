#!/usr/bin/perl -w

use MdmDiscoveryScript;
use strict;

sub strip
{
    my ($line) = @_;
    $line =~ s/^\s+//;
    $line =~ s/\s+$//;
    return $line;
}

sub goUntil {
	(my $fin, my $end_line) = @_;
	while(my $row=<$fin>)
	{
		if ($row =~ /^\Q$end_line/)
		{
			return;
		}
	}
}

my $arglen = @ARGV;

if ($arglen < 2){
	print "Error: usage create_surfaces <path> <last id number>\n";
	print "Use <last id number> <= 0 if you don't know amount of .mol files\n";
	print "Program will read 1.mol 2.mol ... \n";
	print "   until file doesn't exists or <mol file id> > <last id number> > 0\n";
	exit(1);
}

my $path = $ARGV[0];
my $last_id = $ARGV[1]+0;

my $tmpfilename = $path."/temp.wrl";

my $i=1;

while(1)
{
	my $cur_filename = $path."/".$i.".mol";
	
	if (not -e $cur_filename)
	{
		print "No file ".$cur_filename."\n";
		if ($last_id<=0)
		{
			print "Last file number: ".($i-1)."\n";
			last;
		} else{
			next;
		}
	} else {
		print "Process file ".$cur_filename."\n";
	}
	
	my $document = DiscoveryScript::Open({Path=>$cur_filename});
          CreateAndSaveSurfaceChargesAndWDVRadii($document, $i, $path,"");
	$i++;
	
	if ($last_id>0)
	{
		if ($i>$last_id)
		{
			last;
		}
	}
	
}

exit();

s
sub CreateAndSaveSurfaceChargesAndWDVRadii
{

	my ($document, $curr_ind, $path) = @_;
	$path = $path.'/';
        my $molecule = $document->Molecules()->Item(0);

	my $array_of_atoms = $molecule->Atoms; 
	my $surface = $document->CreateSolidSurface( $array_of_atoms, Mdm::surfaceStyleSolvent, False, Mdm::surfaceColorByElectrostaticPotential, 1.4 );	
        my $molsurf = $path."$curr_ind".'.wrl';
	$document->Save($molsurf ,'wrl' ); 

         $document->CalculateCharges();

	open(my $charges_file, '>', $path."$curr_ind".'.ch') ;

	foreach my $atom (@$array_of_atoms)
	{
	    print $charges_file $atom->XYZ->X.' '.$atom->XYZ->Y.' '.$atom->XYZ->Z.' '.$atom->Charge."\n";
	}
	close $charges_file;


	open(my $radii_file, '>', $path."$curr_ind".'.wdv') ;

	foreach my $atom (@$array_of_atoms)
	{
	    print $radii_file $atom->XYZ->X.' '.$atom->XYZ->Y.' '.$atom->XYZ->Z.' '.$atom->VdwRadius."\n";
	}
	
	close $radii_file;
	
	
    #makeSurfaceFile($molsurf, $curr_ind, $path);

	return 1;
}
