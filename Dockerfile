FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

COPY darshan /darshan


#Install dependencies for building Darshan
RUN apt-get update && apt-get install -y \
    gcc \
    make \
    libz-dev \
    autoconf \
    automake \
    libtool \
    bzip2

# Set the library path for macOS
ENV DYLD_FALLBACK_LIBRARY_PATH=/usr/lib

# Install PyDarshan from source
RUN cd /darshan/darshan-util/pydarshan \
    && pip install .

# Build and install Darshan
RUN cd /darshan \
    && ./prepare.sh \
    && autoupdate \
    && aclocal \
    && libtoolize \
    && autoconf \
    && ./configure --prefix=/usr --with-log-path=/usr/share/darshan/logs --with-jobid-env=SLURM_JOBID \
    && make \
    && make install 

ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

ENV DYLD_FALLBACK_LIBRARY_PATH=/usr/lib:$DYLD_FALLBACK_LIBRARY_PATH

# Install graphviz system package and development libraries
RUN apt-get update && apt-get install -y \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    && apt-get clean

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Run convert_to_csv.py when the container launches
# CMD ["python", "convert_to_csv.py"]
# CMD ["python", "read_darshan_logs.py"]
# CMD ["python", "performance_darshan_logs.py"]
# CMD ["python", "performance_csv_uniqe.py"]
# CMD ["python", "graph_gen.py"]
# CMD ["python", "graph_gen2.py"]
# CMD ["python", "graph_gen3.py"]
CMD ["python", "graph_gen4.py"]
# CMD ["python", "graph_represent.py"]
