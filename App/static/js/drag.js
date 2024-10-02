document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const fileList2 = document.getElementById('file-list2')

    console.log('Script loaded');

    if (!dropZone || !fileInput || !fileList) {
        console.error('One or more elements not found');
        return;
    }

    dropZone.addEventListener('click', () => {
        console.log('Drop zone clicked');
        fileInput.click();
    });
    
    dropZone.addEventListener('dragover', (e) => {
        console.log('Drag over');
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        console.log('Drag leave');
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        console.log('File dropped');
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        handleFiles(files);
    });

    fileInput.addEventListener('change', (e) => {
        console.log('File input changed');
        const files = e.target.files;
        handleFiles(files);
    });

    function handleFiles(files) {
        console.log('Handling files:', files.length);
        for (const file of files) {
            uploadFile(file);
        }
    }

    function handleAll(files) {
        console.log('Handling files:', files.length);
        for (const file of files) {
            applyAll(file);
        }
    }


    function uploadFile(file) {
        console.log('Uploading file:', file.name);
        const formData = new FormData();
        formData.append('file', file);
    
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Upload error:', data.error);
            } else {
                console.log('Upload successful:', file.name);
                
                // Populate file-list
                const listItem = document.createElement('p');
                const link = document.createElement('a');
                link.href = data.view_data_url;
                link.textContent = `${file.name} - View Data`;
                link.target = '_blank';
                listItem.appendChild(link);
                fileList.appendChild(listItem);
    
                // Populate file-list2
                const listItem2 = document.createElement('p');
                const link2 = document.createElement('a');
                link2.href = data.all_url;
                link2.textContent = `${file.name} - All`;
                link2.target = '_blank';
                listItem2.appendChild(link2);
                fileList2.appendChild(listItem2);
            }
        })
        .catch(error => console.error('Fetch error:', error));
    }
});