$instruction = $args[0] # extract or repack
$original_jar_file_directory = $args[1] # The directory where the original jar file is located
$original_jar_file_name = $args[2] # The name of the original jar file

$original_jar_file_full_path = "$original_jar_file_directory\$original_jar_file_name"
$extracted_file_directory = "$original_jar_file_directory\jar_extracted_full_files"

if ($instruction -eq "extract") {
    Write-Host "Extracting the jar file $original_jar_file_full_path..."
    & 7z x $original_jar_file_full_path -o"$extracted_file_directory"
    if ($lastExitCode -ne 0) {
        Write-Host -Object "7z extracting the jar file command failed. Exitcode: $exitCode"
        exit $lastExitCode
    }
    Write-Host "Extracted files directory: $extracted_file_directory"

    Write-Host "Removing the original jar file..."
    Remove-Item -Path "$original_jar_file_full_path" -Force
    Write-Host "Removed the original jar file."
}
elseif ($instruction -eq "repack") {
    Write-Host "Removing ESRP's CodeSignSummary file..."
    # It is the summary generated by ESRP tool. It is not needed in the jar file.
    Remove-Item -Path "$extracted_file_directory/CodeSignSummary*.*" -Force
    Write-Host "Removed ESRP's CodeSignSummary file."

    Write-Host "Repacking the jar file from directory $extracted_file_directory..."
    & 7z a "$original_jar_file_full_path" "$extracted_file_directory\*"
    if ($lastExitCode -ne 0) {
        Write-Host -Object "7z repacking the jar file command failed. Exitcode: $exitCode"
        exit $lastExitCode
    }
    Write-Host "Repacked the jar file $original_jar_file_full_path."

    Write-Host "Removing the extracted files..."
    Remove-Item -Path "$extracted_file_directory" -Recurse -Force
    Write-Host "Removed the extracted files."
}
else {
    Write-Host "Invalid instruction: $instruction"
}