ARG ISAAC_BASE_IMAGE
FROM ${ISAAC_BASE_IMAGE}

ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/pathvla-unitree-isaac-live

COPY scripts/isaac_python.sh /tmp/isaac_python.sh
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN bash /tmp/isaac_python.sh -m pip install --no-cache-dir -r /tmp/requirements-dev.txt

COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
