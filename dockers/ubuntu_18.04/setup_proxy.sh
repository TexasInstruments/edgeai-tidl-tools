#!/bin/bash

#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if [ "$USE_PROXY" = "ti" ]; then
	# apt proxy
	if [ ! -f /etc/apt/apt.conf ]; then
		echo "Acquire::http::proxy \"http://webproxy.ext.ti.com:80\";" > /etc/apt/apt.conf
	fi

	# wget proxy
	if [ ! -f ~/.wgetrc ]; then
		cat > ~/.wgetrc << EOF
http_proxy=http://webproxy.ext.ti.com:80
https_proxy=http://webproxy.ext.ti.com:80
ftp_proxy=http://webproxy.ext.ti.com:80
noproxy=ti.com
EOF
	fi

	# pip3 proxy
	if [ ! -f ~/.config/pip/pip.conf ]; then
		mkdir -p ~/.config/pip/
		cat > ~/.config/pip/pip.conf << EOF
[global]
proxy = http://webproxy.ext.ti.com
EOF
	fi

	#git proxy
	cat << END >> ~/.gitconfig
[core]
        gitproxy = none for ti.com
        gitproxy = /home/$USER/git-proxy.sh
[http]
        proxy = http://webproxy.ext.ti.com:80
[https]
        proxy = http://webproxy.ext.ti.com:80
END

   cat << END >> ~/git-proxy.sh
#!/bin/sh
exec /usr/bin/corkscrew webproxy.ext.ti.com 80 $*
END

   chmod +x ~/git-proxy.sh

else
	rm -rf /etc/apt/apt.conf
	rm -rf ~/.wgetrc
	rm -rf ~/.config/pip/pip.conf
	rm -rf ~/.gitconfig ~/git-proxy.sh
fi

